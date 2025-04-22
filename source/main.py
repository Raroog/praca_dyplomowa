import asyncio
import logging
from urllib.parse import urlparse

import yaml
from logging_setup import setup_logging
from scrapers.malpedia_library_downloader import (
    GetMalpediaBibFile,
    ParseMalpediaBibFile,
)
from scrapers.scraper import Scraper

logger = logging.getLogger(__name__)


def load_config(config_path="/home/bartek/Kod/PD/praca_dyplomowa/config.yaml"):
    """Load configuration from YAML file"""
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


async def main():
    config = load_config()
    setup_logging(config)
    base_path = config.get("scraping", {}).get("base_path")

    malpedia_bib = GetMalpediaBibFile()
    bib_file = malpedia_bib.bib_library

    bib_parser = ParseMalpediaBibFile(base_path, bib_file)

    bib_parser.save_dict_as_json()

    for site_data_dict in bib_parser.bib_list_of_dicts[20:30]:
        url = site_data_dict["url"]
        if urlparse(url.replace("www.", "")).hostname in bib_parser.blacklist:
            continue
        title_path = site_data_dict["title"].strip("{}").replace(" ", "_")
        output_path = f"{base_path}/{title_path}"
        scraper = await Scraper.create(url=url, output_path=output_path)

        results = await scraper.scrape_images()
        print(results)
        if results["successes"]:
            metadata_path = f"{output_path}/metadata.json"
            text_w_image_markers_path = f"{output_path}/text_w_image_markers.txt"
            clean_text_path = f"{output_path}/clean_text.txt"

            await scraper.save_metadata(metadata_path)
            await scraper.save_text_w_images(text_w_image_markers_path)
            await scraper.save_clean_text(clean_text_path)


if __name__ == "__main__":
    asyncio.run(main())
