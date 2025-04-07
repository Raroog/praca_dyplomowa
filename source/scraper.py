import json
from pathlib import Path
from urllib.parse import urlparse, urlunsplit
from typing import Any, Dict, List, Tuple

import requests
from bs4 import BeautifulSoup
from trafilatura import extract, fetch_url

import aiohttp        
import aiofiles


class Scraper:
    def __init__(self, url: str, output_path:str) -> None:
        self.url = url
        self.output_path = Path(output_path)
        self.response = requests.get(self.url)
        self.status_code = self.response.status_code
        self.requests_html = self.response.text if self.status_code == 200 else None
        if self.requests_html is not None:
            self.html_soup = BeautifulSoup(self.requests_html, "html.parser")
            self.images_list = self.extract_image_metadata()
            self.text_w_images = self.get_text_with_image_markers()

    def extract_image_metadata(self) -> List[Dict[str, Any]]:
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
    
    def scrape_images(self):
        base_url = urlparse(self.url).netloc

        for img_data in self.images_list:
            img_url = img_data["src"]
            filename = img_url["filename"]
            if not img_data["src"].startswith(('http', 'https')):
                img_url = urlunsplit((base_url, img_url))
            filename = self.output_path / Path(img_url).name

            try:
                async with aiohttp.ClientSession() as session:
                    url = "http://host/file.img"
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            f = await aiofiles.open('/some/file.img', mode='wb')
                            await f.write(await resp.read())
                            await f.close()


            return True


    def save_metadata(self output_path: Path) -> None:
        """Save image metadata to a JSON file."""
        # Remove the BeautifulSoup element reference before saving
        clean_images = []
        for img in self.images_list:
            img_copy = img.copy()
            img_copy.pop("element", None)
            clean_images.append(img_copy)

        with open(output_path, "w") as f:
            json.dump(clean_images, f, indent=2)

