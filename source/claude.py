#!/usr/bin/env python3
"""
Web Scraper with Image Position Tracking

This script combines trafilatura and BeautifulSoup to scrape websites
while preserving the exact positions of images within text content.
"""

import hashlib
import json
import logging
import os
import re
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import requests
import trafilatura
from bs4 import BeautifulSoup, Tag
from trafilatura.settings import use_config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Image marker format: [IMG:id]
IMG_MARKER_PATTERN = r"\[IMG:([0-9a-f]+)\]"


class WebScraper:
    """Class for scraping web content with image position tracking."""

    def __init__(self, output_dir: str = "scraped_content"):
        """
        Initialize the WebScraper.

        Args:
            output_dir: Directory to save images and output files
        """
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")

        # Create output directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        # Configure trafilatura
        self.config = use_config()
        self.config.set("DEFAULT", "extraction_timeout", "30")
        self.config.set("DEFAULT", "MIN_OUTPUT_SIZE", "500")

        # User agent for requests
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def scrape_url(self, url: str) -> Dict[str, Any]:
        """
        Scrape a URL and extract content with image tracking.

        Args:
            url: URL to scrape

        Returns:
            Dictionary with extracted content and metadata
        """
        logger.info(f"Scraping URL: {url}")

        try:
            # Download the webpage
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            html_content = response.text

            # Extract main content using trafilatura
            extracted_content = trafilatura.extract(
                html_content,
                output_format="html",
                include_images=True,
                include_tables=True,
                include_links=True,
                config=self.config,
            )

            if not extracted_content:
                logger.warning(f"No content extracted from {url}")
                return {"error": "No content extracted", "url": url}

            # Process the extracted HTML with BeautifulSoup
            soup = BeautifulSoup(extracted_content, "html.parser")

            # Track and download images
            processed_text, images_metadata = self._process_content(soup, url)

            # Create result object
            result = {
                "url": url,
                "title": self._extract_title(soup),
                "text": processed_text,
                "images": images_metadata,
                "timestamp": self._get_timestamp(),
            }

            return result

        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return {"error": str(e), "url": url}

    def _process_content(
        self, soup: BeautifulSoup, base_url: str
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Process HTML content, track and replace images with markers.

        Args:
            soup: BeautifulSoup object of the HTML content
            base_url: Base URL for resolving relative image paths

        Returns:
            Tuple of (processed text content, list of image metadata)
        """
        images_metadata = []

        # Find all images in the content
        for img in soup.find_all("img"):
            # Generate a unique ID for the image based on src and position
            img_id = self._generate_image_id(img.get("src", ""))

            # Extract image metadata
            img_data = self._extract_image_metadata(img, base_url, img_id)

            if img_data:
                # Download the image
                img_path = self._download_image(img_data["url"], img_id)
                if img_path:
                    img_data["local_path"] = img_path
                    images_metadata.append(img_data)

                    # Replace the image with a marker
                    marker = f"[IMG:{img_id}]"
                    img.replace_with(BeautifulSoup(marker, "html.parser"))

        # Handle lazy-loaded images
        self._process_lazy_loaded_images(soup, base_url, images_metadata)

        # Get the processed text content
        processed_text = str(soup)

        # Clean up the HTML to get plain text with image markers
        processed_text = self._clean_html_to_text(processed_text)

        return processed_text, images_metadata

    def _extract_image_metadata(
        self, img: Tag, base_url: str, img_id: str
    ) -> Optional[Dict[str, str]]:
        """
        Extract metadata from an image tag.

        Args:
            img: BeautifulSoup Tag object for the image
            base_url: Base URL for resolving relative paths
            img_id: Unique identifier for the image

        Returns:
            Dictionary with image metadata or None if invalid
        """
        src = img.get("src", "")
        if not src or src.startswith("data:"):
            # Check for data-src (lazy loading)
            src = img.get("data-src", "")
            if not src or src.startswith("data:"):
                return None

        # Resolve relative URLs
        if not src.startswith(("http://", "https://")):
            src = urllib.parse.urljoin(base_url, src)

        return {
            "id": img_id,
            "url": src,
            "alt": img.get("alt", ""),
            "title": img.get("title", ""),
            "width": img.get("width", ""),
            "height": img.get("height", ""),
        }

    def _generate_image_id(self, src: str) -> str:
        """
        Generate a unique ID for an image based on its source URL.

        Args:
            src: Image source URL

        Returns:
            Hexadecimal hash ID
        """
        hash_obj = hashlib.md5(src.encode())
        return hash_obj.hexdigest()[:12]

    def _download_image(self, img_url: str, img_id: str) -> Optional[str]:
        """
        Download an image and save it to the images directory.

        Args:
            img_url: URL of the image
            img_id: Unique identifier for the image

        Returns:
            Local path to the saved image or None if download failed
        """
        try:
            # Determine file extension from URL
            parsed_url = urllib.parse.urlparse(img_url)
            path = parsed_url.path
            extension = os.path.splitext(path)[1].lower()

            # Default to .jpg if no extension
            if not extension:
                extension = ".jpg"

            # Create filename
            filename = f"{img_id}{extension}"
            local_path = os.path.join(self.images_dir, filename)

            # Download and save the image
            response = requests.get(img_url, headers=self.headers, timeout=30)
            response.raise_for_status()

            with open(local_path, "wb") as f:
                f.write(response.content)

            return local_path

        except Exception as e:
            logger.error(f"Error downloading image {img_url}: {str(e)}")
            return None

    def _process_lazy_loaded_images(
        self, soup: BeautifulSoup, base_url: str, images_metadata: List[Dict[str, str]]
    ):
        """
        Process lazy-loaded images that might not be in img tags.

        Args:
            soup: BeautifulSoup object
            base_url: Base URL for resolving relative paths
            images_metadata: List to append image metadata to
        """
        # Look for common lazy-loading patterns
        for tag in soup.find_all(["div", "figure"]):
            data_src = tag.get("data-src", "")
            data_img = tag.get("data-img", "")
            data_original = tag.get("data-original", "")

            src = data_src or data_img or data_original
            if src and not src.startswith("data:"):
                # Found a lazy-loaded image
                img_id = self._generate_image_id(src)

                # Resolve relative URL
                if not src.startswith(("http://", "https://")):
                    src = urllib.parse.urljoin(base_url, src)

                # Create metadata
                img_data = {
                    "id": img_id,
                    "url": src,
                    "alt": tag.get("alt", "") or tag.get("data-alt", ""),
                    "title": tag.get("title", "") or tag.get("data-title", ""),
                    "width": tag.get("data-width", ""),
                    "height": tag.get("data-height", ""),
                }

                # Download the image
                img_path = self._download_image(src, img_id)
                if img_path:
                    img_data["local_path"] = img_path
                    images_metadata.append(img_data)

                    # Replace with marker
                    marker = f"[IMG:{img_id}]"
                    tag.replace_with(BeautifulSoup(marker, "html.parser"))

    def _clean_html_to_text(self, html_content: str) -> str:
        """
        Clean HTML to plain text while preserving image markers.

        Args:
            html_content: HTML content with image markers

        Returns:
            Cleaned text with image markers preserved
        """
        # Extract image markers first
        markers = []
        for match in re.finditer(IMG_MARKER_PATTERN, html_content):
            markers.append(match.group(0))

        # Parse HTML
        soup = BeautifulSoup(html_content, "html.parser")

        # Preserve paragraph structure
        for p in soup.find_all("p"):
            p.insert_after(BeautifulSoup("\n\n", "html.parser"))

        # Preserve list items
        for li in soup.find_all("li"):
            li.insert_before(BeautifulSoup("• ", "html.parser"))
            li.insert_after(BeautifulSoup("\n", "html.parser"))

        # Get text content
        text = soup.get_text()

        # Clean up extra whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        text = text.strip()

        # Re-insert markers that might have been lost in the cleaning
        for marker in markers:
            if marker not in text:
                # Check if the marker ID exists in the cleaned text
                marker_id = re.search(IMG_MARKER_PATTERN, marker).group(1)
                pattern = r"\[IMG:[0-9a-f]*" + marker_id + r"[0-9a-f]*\]"
                if not re.search(pattern, text):
                    # If not found, add it at a reasonable position (end of a paragraph)
                    text = text.replace("\n\n", f"\n\n{marker}\n\n", 1)

        return text

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """
        Extract title from BeautifulSoup object.

        Args:
            soup: BeautifulSoup object

        Returns:
            Title text
        """
        title_tag = soup.find("title")
        if title_tag:
            return title_tag.get_text().strip()

        h1_tag = soup.find("h1")
        if h1_tag:
            return h1_tag.get_text().strip()

        return "Untitled Document"

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime

        return datetime.now().isoformat()

    def save_result(self, result: Dict[str, Any], filename: str = None) -> str:
        """
        Save the scraping result to a JSON file.

        Args:
            result: Dictionary with scraping results
            filename: Optional filename override

        Returns:
            Path to the saved file
        """
        if not filename:
            # Generate filename from URL
            url_parts = urllib.parse.urlparse(result.get("url", "unknown"))
            domain = url_parts.netloc.replace(".", "_")
            timestamp = result.get("timestamp", "").replace(":", "-").split(".")[0]
            filename = f"{domain}_{timestamp}.json"

        output_path = os.path.join(self.output_dir, filename)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved result to {output_path}")
        return output_path

    def reconstruct_document(
        self, result: Dict[str, Any], output_format: str = "html"
    ) -> str:
        """
        Reconstruct the document with images in their original positions.

        Args:
            result: Dictionary with scraping results
            output_format: Output format (html or markdown)

        Returns:
            Reconstructed document in the specified format
        """
        text = result.get("text", "")
        images = result.get("images", [])

        # Create mapping of image IDs to metadata
        img_map = {img["id"]: img for img in images}

        if output_format == "html":
            # Replace markers with HTML img tags
            for match in re.finditer(IMG_MARKER_PATTERN, text):
                marker = match.group(0)
                img_id = match.group(1)

                if img_id in img_map:
                    img_data = img_map[img_id]
                    img_tag = (
                        f'<img src="{img_data["local_path"]}" alt="{img_data["alt"]}" '
                    )
                    if img_data["width"]:
                        img_tag += f'width="{img_data["width"]}" '
                    if img_data["height"]:
                        img_tag += f'height="{img_data["height"]}" '
                    img_tag += "/>"
                    text = text.replace(marker, img_tag)

            # Wrap in HTML structure
            html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{result.get("title", "Untitled")}</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
        img {{ max-width: 100%; height: auto; display: block; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>{result.get("title", "Untitled")}</h1>
    <div class="content">
        {text.replace("\n\n", "</p><p>")}
    </div>
    <div class="metadata">
        <p>Source: <a href="{result.get("url", "#")}">{result.get("url", "Unknown")}</a></p>
        <p>Scraped on: {result.get("timestamp", "Unknown")}</p>
    </div>
</body>
</html>"""
            return html

        elif output_format == "markdown":
            # Replace markers with Markdown image syntax
            for match in re.finditer(IMG_MARKER_PATTERN, text):
                marker = match.group(0)
                img_id = match.group(1)

                if img_id in img_map:
                    img_data = img_map[img_id]
                    md_img = f"![{img_data['alt']}]({img_data['local_path']})"
                    text = text.replace(marker, md_img)

            # Add markdown header
            md = f"# {result.get('title', 'Untitled')}\n\n"
            md += text
            md += f"\n\nSource: {result.get('url', 'Unknown')}\n"
            md += f"Scraped on: {result.get('timestamp', 'Unknown')}\n"

            return md

        else:
            raise ValueError(f"Unsupported output format: {output_format}")


def scrape_website(
    url: str, output_dir: str = "scraped_content", output_format: str = "html"
) -> Dict[str, Any]:
    """
    Scrape a website and save the results.

    Args:
        url: URL to scrape
        output_dir: Directory to save output
        output_format: Format for reconstruction (html or markdown)

    Returns:
        Dictionary with results and paths
    """
    scraper = WebScraper(output_dir=output_dir)

    # Scrape the URL
    result = scraper.scrape_url(url)

    if "error" in result:
        return result

    # Save the JSON result
    json_path = scraper.save_result(result)

    # Reconstruct the document
    reconstructed = scraper.reconstruct_document(result, output_format=output_format)

    # Save the reconstructed document
    recon_filename = os.path.basename(json_path).replace(".json", f".{output_format}")
    recon_path = os.path.join(output_dir, recon_filename)

    with open(recon_path, "w", encoding="utf-8") as f:
        f.write(reconstructed)

    # Add paths to result
    result["json_path"] = json_path
    result["reconstructed_path"] = recon_path
    result["output_format"] = output_format

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Web scraper with image position tracking"
    )
    parser.add_argument("url", help="URL to scrape")
    parser.add_argument(
        "--output-dir", "-o", default="scraped_content", help="Output directory"
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["html", "markdown"],
        default="html",
        help="Output format",
    )

    args = parser.parse_args()

    result = scrape_website(args.url, args.output_dir, args.format)

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Successfully scraped {result['url']}")
        print(f"Title: {result['title']}")
        print(f"Images found: {len(result['images'])}")
        print(f"JSON saved to: {result['json_path']}")
        print(f"Reconstructed document saved to: {result['reconstructed_path']}")
