import asyncio
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunsplit

import aiofiles
import aiohttp
from bs4 import BeautifulSoup
from trafilatura import extract

logger = logging.getLogger(__name__)


class Scraper:
    def __init__(self, url: str, output_path: str) -> None:
        self.url = url
        self.output_path = Path(output_path)
        self.requests_html = None
        self.html_soup = None
        self.text_from_html = None
        self.images_list = []
        self.text_w_images = ""
        self.status_code = None

    @classmethod
    async def create(cls, url: str, output_path: str):
        self = cls(url, output_path)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.url) as response:
                    self.status_code = response.status
                    if self.status_code == 200:
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
        except Exception as e:
            logger.error("Unexpected error when accessing %s: %s", self.url, str(e))
            self.status_code = 0
            self.requests_html = None

        if self.requests_html is not None:
            logger.info("Successfully downloaded site HTML")
            self.html_soup = BeautifulSoup(self.requests_html, "html.parser")
            self.text_from_html = extract(self.requests_html, output_format="txt")
            self.images_list = self.extract_image_metadata()
            self.text_w_images = self.get_text_with_image_markers()
        return self

    def extract_image_metadata(self) -> list[dict[str, Any]]:
        """Extract metadata for all images in the HTML."""
        images = []
        for i, img in enumerate(self.html_soup.find_all("img")):
            image_id = f"{i + 1}"
            filename = Path(img.get("src", "-")).name
            images.append(
                {
                    "id": image_id,
                    "filename": filename[:100],
                    "src": img.get("src", "-"),
                    "element": img,  # Store reference to original element
                }
            )
        logger.info("Extracted image data")
        return images

    def get_text_with_image_markers(self) -> str:
        """
        Extract text from HTML with image markers inserted at appropriate positions.
        """
        # Make a copy of the soup to avoid modifying the original
        soup_copy = BeautifulSoup(self.requests_html, "html.parser")

        # Replace each image with a marker
        for img_data in self.images_list:
            img_element = soup_copy.find(
                "img",
                src=img_data["src"],
            )
            if img_element:
                marker = soup_copy.new_string(f"[IMG:{img_data['id']}]")
                img_element.replace_with(marker)

        # Extract text, removing excessive whitespace
        text = " ".join(soup_copy.get_text().split())
        logger.info("Constructed text with image markers from HTML")
        return text

    async def scrape_images(self):
        base_url_scheme = urlparse(self.url).scheme
        base_url_netloc = urlparse(self.url).netloc

        async with aiohttp.ClientSession() as session:
            download_tasks = []
            for img_data in self.images_list:
                src = img_data["src"]
                if not src.startswith(("http", "https")):
                    src = urlunsplit((base_url_scheme, base_url_netloc, src, "", ""))
                filename = self.output_path / Path(img_data["filename"])
                task = self.download_image(session, src, filename)
                download_tasks.append(task)
            results = await asyncio.gather(*download_tasks, return_exceptions=True)
            succcesses = sum(1 for r in results if r is True)
            failures = sum(1 for r in results if r is not True)
            return {
                "total": len(download_tasks),
                "successes": succcesses,
                "failures": failures,
            }

    async def download_image(self, session, url, filename):
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    filename.parent.mkdir(exist_ok=True)
                    async with aiofiles.open(f"{filename}", mode="wb") as f:
                        await f.write(await response.read())
                    logger.info(
                        "Successfully downloaded image: %s, from url: %s",
                        filename,
                        url,
                    )
                    return True
                return False
        except Exception as e:
            logger.exception("%s: %s", url, e)
            return False

    async def save_metadata(self, output_path: Path | str) -> None:
        """Save image metadata to a JSON file."""
        # Remove the BeautifulSoup element reference before saving
        clean_images = []
        for img in self.images_list:
            img_copy = img.copy()
            img_copy.pop("element", None)
            clean_images.append(img_copy)

        async with aiofiles.open(output_path, "w") as f:
            json_str = json.dumps(clean_images, indent=2)
            await f.write(json_str)
        logger.info("Saved image metadata at %s", output_path)

    async def save_clean_text(self, text_path):
        async with aiofiles.open(text_path, "w") as f:
            await f.write(self.text_from_html)
        logger.info("Saved clean text at %s", text_path)

    async def save_text_w_images(self, text_path):
        async with aiofiles.open(text_path, "w") as f:
            await f.write(self.text_w_images)
        logger.info("Saved text with image markers at %s", text_path)
