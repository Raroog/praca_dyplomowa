import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import aiofiles
import tiktoken
from dotenv import dotenv_values
from langfuse.openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = dotenv_values(".env")

OPENAI_API_KEY = config.get("OPENAI_API_KEY")

os.environ["LANGFUSE_PUBLIC_KEY"] = config.get("LANGFUSE_PUBLIC_KEY")
os.environ["LANGFUSE_SECRET_KEY"] = config.get("LANGFUSE_SECRET_KEY")
os.environ["LANGFUSE_HOST"] = config.get("LANGFUSE_HOST")


class TokenAwareRateLimiter:
    """Rate limiter that respects OpenAI Tier 2 RPM and TPM limits."""

    def __init__(
        self,
        requests_per_minute: int = 4_500,  # Conservative for Tier 2 (5K max)
        tokens_per_minute: int = 1_800_000,  # Conservative for Tier 2 (2M max)
        max_concurrent: int = 50,
    ):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.request_interval = 60.0 / requests_per_minute
        self.semaphore = asyncio.Semaphore(max_concurrent)

        self.last_request_time = 0.0
        self.token_usage_window: list[tuple[float, int]] = []
        self.lock = asyncio.Lock()

    def _clean_token_window(self, current_time: float) -> None:
        """Remove token usage entries older than 1 minute."""
        cutoff = current_time - 60.0
        self.token_usage_window = [
            (ts, tokens) for ts, tokens in self.token_usage_window if ts > cutoff
        ]

    def _get_current_token_usage(self, current_time: float) -> int:
        """Get total tokens used in the last minute."""
        self._clean_token_window(current_time)
        return sum(tokens for _, tokens in self.token_usage_window)

    async def acquire(self, estimated_tokens: int) -> None:
        """Wait until we can make another request without exceeding limits."""
        async with self.lock:
            while True:
                now = asyncio.get_event_loop().time()

                time_since_last = now - self.last_request_time
                if time_since_last < self.request_interval:
                    wait_time = self.request_interval - time_since_last
                    logger.debug("RPM limit: waiting %.2fs", wait_time)
                    await asyncio.sleep(wait_time)
                    continue

                current_usage = self._get_current_token_usage(now)
                if current_usage + estimated_tokens > self.tokens_per_minute:
                    wait_time = 5.0
                    logger.warning(
                        "TPM limit: %d tokens used, waiting %.2fs",
                        current_usage,
                        wait_time,
                    )
                    await asyncio.sleep(wait_time)
                    continue

                self.last_request_time = now
                self.token_usage_window.append((now, estimated_tokens))
                break


def count_tokens(
    messages: list[dict[str, str]], model: str = "gpt-5-2025-08-07"
) -> int:
    """Estimate token count for messages using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    for message in messages:
        num_tokens += 4
        num_tokens += len(encoding.encode(message["content"]))
    num_tokens += 2
    print(num_tokens)

    return num_tokens


async def save_result(
    result: dict[str, Any],
    original_path: Path,
    output_dir: Path,
) -> bool:
    """Save processing result to disk asynchronously."""
    if not result["success"]:
        logger.error("Skipping save for failed result: %s", original_path.name)
        return False

    try:
        relative_path = original_path.relative_to(original_path.parent.parent)
        output_path = output_dir / relative_path.parent / f"LLM_{original_path.name}"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        content = result["response"].choices[0].message.content

        async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
            await f.write(content)

        logger.info("Saved result to %s", output_path)
        return True

    except Exception as e:
        logger.error("Failed to save result for %s: %s", original_path.name, e)
        return False


async def process_text_with_retry(
    messages: list[dict[str, str]],
    client: AsyncOpenAI,
    rate_limiter: TokenAwareRateLimiter,
    original_path: Path,
    output_dir: Path,
    model: str = "gpt-5-2025-08-07",
    max_retries: int = 3,
) -> dict[str, Any]:
    """Process a single text with rate limiting, retry logic, and async saving."""
    estimated_tokens = count_tokens(messages, model)
    logger.debug("Estimated tokens for %s: %d", original_path.name, estimated_tokens)

    for attempt in range(max_retries):
        try:
            async with rate_limiter.semaphore:
                await rate_limiter.acquire(estimated_tokens)

                response = await client.chat.completions.create(
                    messages=messages,
                    model=model,
                    reasoning_effort="low",
                    verbosity="low",
                )

                result = {"success": True, "response": response, "error": None}
                await save_result(result, original_path, output_dir)
                return result

        except Exception as e:
            error_msg = str(e)

            if "rate" in error_msg.lower() or "429" in error_msg:
                wait_time = (2**attempt) * 10
                logger.warning(
                    "Rate limit hit for %s, waiting %ds (retry %d/%d)",
                    original_path.name,
                    wait_time,
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error("Error processing %s: %s", original_path.name, error_msg)
                return {"success": False, "response": None, "error": error_msg}

    return {"success": False, "response": None, "error": "Max retries exceeded"}


async def process_batch(
    texts_data: list[tuple[list[dict[str, str]], Path]],
    client: AsyncOpenAI,
    rate_limiter: TokenAwareRateLimiter,
    output_dir: Path,
    batch_size: int = 50,
) -> list[dict[str, Any]]:
    """Process texts in batches with rate limiting."""
    results = []
    total = len(texts_data)

    for i in range(0, total, batch_size):
        batch = texts_data[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        logger.info(
            "Processing batch %d/%d (%d texts)",
            batch_num,
            total_batches,
            len(batch),
        )

        tasks = [
            process_text_with_retry(messages, client, rate_limiter, path, output_dir)
            for messages, path in batch
        ]

        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

        success_count = sum(1 for r in batch_results if r["success"])
        logger.info(
            "Batch %d completed: %d/%d successful",
            batch_num,
            success_count,
            len(batch),
        )

    return results


async def main() -> None:
    """Main processing function."""
    system_message = """You are a cybersecurity text processor. Clean up formatting issues in scraped blog posts and OCR text from malware analysis materials.
For scraped content: restructure single-line text into readable paragraphs.
For OCR: fix character misrecognition in code, commands, and URLs while maintaining technical accuracy.
Do not interpret, summarize, or modify the actual cybersecurity content - only correct formatting and obvious transcription errors."""

    texts_path = Path(
        "/home/bartek/Kod/PD/praca_dyplomowa/dane/texts/ocr_enriched_texts"
    )
    texts_paths = list(texts_path.glob("**/clean_text*.txt"))
    text_dirs = {path.parts[-1] for path in texts_path.glob("*")}

    output_dir = Path("/home/bartek/Kod/PD/praca_dyplomowa/dane/texts/cleaned_texts")
    output_dirs = {path.parts[-1] for path in output_dir.glob("*")}

    dirty_dirs = text_dirs - output_dirs

    dirty_paths = [path for path in texts_paths if path.parts[-2] in dirty_dirs]

    logger.info("Found %d files to process", len(dirty_paths))

    texts_data = []
    for text_path in dirty_paths:
        async with aiofiles.open(text_path, "r", encoding="utf-8") as file:
            input_text = await file.read()

        user_message = f"""Correct the formatting of this cybersecurity text for RAG embedding.
For scraped blog posts: convert single-line text to properly structured paragraphs with line breaks.
For OCR text (marked with XML tags): fix character recognition errors in code, URLs, and technical terms, then format appropriately.
Remove any duplicated paragraphs or repeated sections.
Preserve all technical content and terminology exactly. Do not add explanations or summaries.

Text to correct:
{input_text}"""

        messages = [
            {"role": "developer", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        texts_data.append((messages, text_path))

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    rate_limiter = TokenAwareRateLimiter(
        requests_per_minute=4_500,
        tokens_per_minute=1_800_000,
        max_concurrent=50,
    )

    logger.info("Starting processing with OpenAI gpt-5-2025-08-07 (Tier 3 limits)...")
    results = await process_batch(
        texts_data,
        client,
        rate_limiter,
        output_dir,
        batch_size=50,
    )

    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    logger.info("Processing complete: %d successful, %d failed", successful, failed)
    return results


if __name__ == "__main__":
    asyncio.run(main())
