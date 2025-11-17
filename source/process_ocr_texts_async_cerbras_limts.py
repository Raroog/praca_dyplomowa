import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import aiofiles
from dotenv import dotenv_values
from langfuse.openai import AsyncOpenAI, openai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = dotenv_values(".env")

CEREBRAS_API_KEY = config.get("CEREBRAS_API_KEY")

os.environ["LANGFUSE_PUBLIC_KEY"] = config.get("LANGFUSE_PUBLIC_KEY")
os.environ["LANGFUSE_SECRET_KEY"] = config.get("LANGFUSE_SECRET_KEY")
os.environ["LANGFUSE_HOST"] = config.get("LANGFUSE_HOST")


class RateLimiter:
    """Rate limiter that respects Cerebras API quotas."""

    def __init__(
        self,
        requests_per_minute: int = 10,
        max_concurrent: int = 5,
    ):
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.last_request_time = 0.0
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until we can make another request."""
        async with self.lock:
            now = asyncio.get_event_loop().time()
            time_since_last = now - self.last_request_time

            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                await asyncio.sleep(wait_time)

            self.last_request_time = asyncio.get_event_loop().time()


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
        # Create output path mirroring input structure
        relative_path = original_path.relative_to(original_path.parent.parent)
        output_path = output_dir / relative_path.parent / f"LLM_{original_path.name}"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract content from response
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
    rate_limiter: RateLimiter,
    original_path: Path,
    output_dir: Path,
    model: str = "gpt-oss-120b",
    name: str = "cerebras-text-cleanup",
    max_retries: int = 3,
) -> dict[str, Any]:
    """Process a single text with rate limiting, retry logic, and async saving."""

    for attempt in range(max_retries):
        try:
            async with rate_limiter.semaphore:
                await rate_limiter.acquire()

                response = await client.chat.completions.create(
                    messages=messages,
                    model=model,
                    name=name,
                    metadata={"provider": "cerebras"},
                )

                result = {"success": True, "response": response, "error": None}

                # Save result immediately after successful processing
                await save_result(result, original_path, output_dir)

                return result

        except Exception as e:
            error_msg = str(e)

            if "429" in error_msg or "rate" in error_msg.lower():
                wait_time = (2**attempt) * 30
                logger.warning(
                    "Rate limit hit, waiting %ds before retry %d/%d",
                    wait_time,
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error("Error processing text: %s", error_msg)
                return {"success": False, "response": None, "error": error_msg}

    return {"success": False, "response": None, "error": "Max retries exceeded"}


async def process_batch(
    texts_data: list[tuple[list[dict[str, str]], Path]],
    client: AsyncOpenAI,
    rate_limiter: RateLimiter,
    output_dir: Path,
    batch_size: int = 10,
) -> list[dict[str, Any]]:
    """Process texts in batches with rate limiting."""

    results = []
    total = len(texts_data)

    for i in range(0, total, batch_size):
        batch = texts_data[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        logger.info(
            "Processing batch %d/%d (%d texts)", batch_num, total_batches, len(batch)
        )

        tasks = [
            process_text_with_retry(messages, client, rate_limiter, path, output_dir)
            for messages, path in batch
        ]

        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

        success_count = sum(1 for r in batch_results if r["success"])
        logger.info(
            "Batch %d completed: %d/%d successful", batch_num, success_count, len(batch)
        )

    return results


async def main() -> None:
    """Main processing function."""

    system_message = """
You are a cybersecurity text processor. Clean up formatting issues in scraped blog posts and OCR text from malware analysis materials.
For scraped content: restructure single-line text into readable paragraphs.
For OCR: fix character misrecognition in code, commands, and URLs while maintaining technical accuracy.
Do not interpret, summarize, or modify the actual cybersecurity content - only correct formatting and obvious transcription errors.
"""

    texts_path = Path(
        "/home/bartek/Kod/PD/praca_dyplomowa/dane/texts/ocr_enriched_texts"
    )

    texts_paths = list(texts_path.glob("**/clean_text*.txt"))

    output_dir = Path("/home/bartek/Kod/PD/praca_dyplomowa/dane/texts/cleaned_texts")

    logger.info("Found %d files to process", len(texts_paths))

    # Prepare messages with original paths
    texts_data = []
    for text_path in texts_paths:
        with text_path.open("r", encoding="utf-8") as file:
            input_text = file.read()

        user_message = f"""
Correct the formatting of this cybersecurity text for RAG embedding.
For scraped blog posts: convert single-line text to properly structured paragraphs with line breaks.
For OCR text (marked with XML tags): fix character recognition errors in code, URLs, and technical terms, then format appropriately.
Remove any duplicated paragraphs or repeated sections.
Preserve all technical content and terminology exactly. Do not add explanations or summaries.

Text to correct:
{input_text}
"""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        texts_data.append((messages, text_path))

    # Initialize client and rate limiter
    client = AsyncOpenAI(
        base_url="https://api.cerebras.ai/v1",
        api_key=CEREBRAS_API_KEY,
    )

    rate_limiter = RateLimiter(
        requests_per_minute=10,
        max_concurrent=5,
    )

    # Process all texts
    logger.info("Starting processing...")
    results = await process_batch(
        texts_data, client, rate_limiter, output_dir, batch_size=10
    )

    # Summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    logger.info("Processing complete: %d successful, %d failed", successful, failed)

    return results


if __name__ == "__main__":
    results = asyncio.run(main())
