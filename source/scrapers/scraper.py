import asyncio
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunsplit

import aiofiles
import aiohttp
from bs4 import BeautifulSoup
from trafilatura import extract


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

        async with aiohttp.ClientSession() as session:
            async with session.get(self.url) as response:
                self.status_code = response.status
                self.requests_html = (
                    await response.text() if self.status_code == 200 else None
                )

        if self.requests_html is not None:
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
                    "filename": filename,
                    "src": img.get("src", "-"),
                    "element": img,  # Store reference to original element
                }
            )
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
                "img", src=img_data["src"], alt=img_data["alt"]
            )
            if img_element:
                marker = soup_copy.new_string(f"[IMG:{img_data['id']}]")
                img_element.replace_with(marker)

        # Extract text, removing excessive whitespace
        text = " ".join(soup_copy.get_text().split())
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
                    filename.parent.mkdir(parents=True, exists_ok=True)
                    async with aiofiles.open(filename, mode="wb") as f:
                        await f.write(await response.read())
                    return True
                return False
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False

    async def save_metadata(self, output_path: Path) -> None:
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
