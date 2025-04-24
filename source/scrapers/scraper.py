import asyncio
import base64
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote, urlparse, urlunsplit

import aiofiles
import aiohttp
from bs4 import BeautifulSoup
from trafilatura import extract

logger = logging.getLogger(__name__)


class Scraper:
    def __init__(
        self,
        url: str,
        output_path: str,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        self.url = url
        self.output_path = Path(output_path)
        self.session = session
        self.requests_html = None
        self.html_soup = None
        self.text_from_html = None
        self.images_list = []
        self.text_w_images = ""
        self.status_code = None

    @classmethod
    async def create(
        cls,
        url: str,
        output_path: str,
        session: Optional[aiohttp.ClientSession] = None,
        timeout: int = 60,
    ):
        self = cls(url, output_path, session)
        provided_session = session is not None
        if not provided_session:
            session = aiohttp.ClientSession()
        try:
            async with session.get(
                self.url, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                self.status_code = response.status
                if self.status_code == 200:
                    logger.info("Started downloading data from url %s", self.url)
                    self.requests_html = await response.text()

                else:
                    logger.warning(
                        "Failed to download HTML: Status code %s for URL %s",
                        self.status_code,
                        self.url,
                    )
                    self.requests_html = None
        except aiohttp.ClientError as e:
            logger.error("HTTP error when accessing %s: %s", self.url, str(e))
            self.status_code = 0
            self.requests_html = None
        except asyncio.TimeoutError:
            logger.error("Timeout when accessing %s", self.url)
            self.status_code = 408  # Request Timeout
            self.requests_html = None
        except Exception as e:
            logger.error("Unexpected error when accessing %s: %s", self.url, str(e))
            self.status_code = 0
            self.requests_html = None
        finally:
            if not provided_session and session:
                await session.close()

        if self.requests_html is not None:
            try:
                self.html_soup = BeautifulSoup(self.requests_html, "html.parser")
                self.text_from_html = extract(self.requests_html, output_format="txt")
                self.images_list = self.extract_image_metadata()
                self.text_w_images = self.get_text_with_image_markers()
                logger.info("Successfully processed HTML content from %s", self.url)
            except Exception as e:
                logger.error("Error processing HTML from %s: %s", self.url, str(e))

        return self

    def extract_image_metadata(self) -> list[dict[str, Any]]:
        """Extract metadata for all images in the HTML."""
        images = []
        for i, img in enumerate(
            self.html_soup.find_all("img", limit=20),
        ):
            image_id = f"{i + 1}"
            if not img.get("src"):
                continue

            src = self.decode_image_url(img.get("src", "-"))
            filename = Path(src).name
            if "icon" in src.lower() or "logo" in src.lower():
                continue
            images.append(
                {
                    "id": image_id,
                    "filename": filename[:100],
                    "src": src,
                    "index": i,
                }
            )
        logger.info("Extracted %d images from %s", len(images), self.url)
        return images

    def decode_image_url(self, src):
        # Check if it's a data URL with Base64 encoded SVG
        if src.startswith("data:image/svg+xml;base64,"):
            try:
                # Extract the Base64 part
                base64_data = src.split("base64,")[1]

                # Decode the SVG
                svg_content = base64.b64decode(base64_data).decode("utf-8")

                # Use regex to extract data-u attribute (more efficient than parsing full XML)
                match = re.search(r'data-u=["\']([^"\']+)["\']', svg_content)
                if match:
                    original_url = unquote(match.group(1))
                    return original_url

                # If regex fails, try parsing as XML
                svg_soup = BeautifulSoup(svg_content, "xml")
                svg_element = svg_soup.find("svg")
                if svg_element and svg_element.has_attr("data-u"):
                    return unquote(svg_element["data-u"])
            except Exception:
                # If decoding fails, just return the original
                pass

        return src

    def get_text_with_image_markers(self) -> str:
        """
        Extract text from HTML with image markers inserted at appropriate positions.
        """
        # Make a copy of the soup to avoid modifying the original
        soup_copy = BeautifulSoup(self.requests_html, "html.parser")

        # Replace each image with a marker
        for img_data in self.images_list:
            img_element = None
            src = img_data["src"]

            img_element = soup_copy.find("img", src=src)

            if not img_element:
                for img in soup_copy.find_all("img"):
                    if img.get("src") and src in img.get("src"):
                        img_element = img
                        break

            if img_element:
                marker = soup_copy.new_string(f"[IMG:{img_data['id']}]")
                img_element.replace_with(marker)

        # Extract text, removing excessive whitespace
        text = " ".join(soup_copy.get_text().split())
        logger.info("Constructed text with image markers from HTML")
        return text

    async def scrape_images(self):
        if not self.images_list:
            return {"total": 0, "successes": 0, "failures": 0}

        base_url_scheme = urlparse(self.url).scheme
        base_url_netloc = urlparse(self.url).netloc

        provided_session = self.session is not None
        session = self.session or aiohttp.ClientSession()

        try:
            semaphore = asyncio.Semaphore(10)

            async def download_with_semaphore(img_data):
                async with semaphore:
                    src = img_data["src"]
                    if not src.startswith(("http", "https")):
                        src = urlunsplit(
                            (base_url_scheme, base_url_netloc, src, "", "")
                        )
                    filename = self.output_path / Path(img_data["filename"])
                    return await self.download_image(session, src, filename)

            # Create tasks for all images
            download_tasks = [
                download_with_semaphore(img_data) for img_data in self.images_list
            ]

            # Execute downloads with timeout
            results = await asyncio.gather(*download_tasks, return_exceptions=True)

            successes = sum(1 for r in results if r is True)
            failures = sum(1 for r in results if r is not True)

            return {
                "total": len(download_tasks),
                "successes": successes,
                "failures": failures,
            }
        finally:
            # Only close the session if we created it
            if not provided_session:
                await session.close()

    async def download_image(self, session, url, filename):
        try:
            timeout = aiohttp.ClientTimeout(total=20)  # 20 seconds timeout
            async with session.get(url, timeout=timeout) as response:
                if response.status == 200:
                    filename.parent.mkdir(exist_ok=True)

                    content_type = response.headers.get("Content-Type", "")
                    if not content_type.startswith("image/"):
                        logger.warning(
                            f"Skipping non-image content type: {content_type} for {url}"
                        )
                        return False

                    async with aiofiles.open(f"{filename}", mode="wb") as f:
                        await f.write(await response.read())

                    logger.info(
                        "Successfully downloaded image: %s, from url: %s",
                        filename,
                        url,
                    )
                    return True
                return False
        except asyncio.TimeoutError:
            logger.warning(f"Timeout downloading image: {url}")
            return False
        except Exception as e:
            logger.exception("%s: %s", url, e)
            return False

    async def save_metadata(self, output_path: Path | str) -> None:
        """Save image metadata to a JSON file."""
        # Remove the BeautifulSoup element reference before saving
        clean_images = []
        for img in self.images_list:
            img_copy = {k: v for k, v in img.items() if k != "element"}
            clean_images.append(img_copy)

        # Ensure parent directory exists
        Path(output_path).parent.mkdir(exist_ok=True, parents=True)

        # Write file with proper error handling
        try:
            async with aiofiles.open(output_path, "w") as f:
                json_str = json.dumps(clean_images, indent=2)
                await f.write(json_str)
            logger.info("Saved image metadata for url %s at %s", self.url, output_path)
        except Exception as e:
            logger.error("Error saving metadata for %s: %s", self.url, str(e))

    async def save_clean_text(self, text_path):
        try:
            # Ensure parent directory exists
            Path(text_path).parent.mkdir(exist_ok=True, parents=True)

            async with aiofiles.open(text_path, "w") as f:
                await f.write(self.text_from_html or "")
            logger.info("Saved clean text for url %s at %s", self.url, text_path)
        except Exception as e:
            logger.error("Error saving clean text for %s: %s", self.url, str(e))

    async def save_text_w_images(self, text_path):
        try:
            # Ensure parent directory exists
            Path(text_path).parent.mkdir(exist_ok=True, parents=True)

            async with aiofiles.open(text_path, "w") as f:
                await f.write(self.text_w_images or "")
            logger.info(
                "Saved text with image markers for url %s at %s", self.url, text_path
            )
        except Exception as e:
            logger.error("Error saving text with images for %s: %s", self.url, str(e))
