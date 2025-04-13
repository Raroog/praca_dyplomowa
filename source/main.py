# from .malpedia_library_downloader import GetMalpediaBibFile, ParseMalpediaBibFile
import asyncio
import logging
from pathlib import Path

import yaml
from logging_setup import setup_logging
from scrapers.malpedia_library_downloader import (
    GetMalpediaBibFile,
    ParseMalpediaBibFile,
)

# from .scrapers.scraper import Scraper

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
    print("aaaaa")

    config = load_config()
    setup_logging(config)
    base_path = config.get("scraping", {}).get("base_path")

    malpedia_bib = GetMalpediaBibFile()
    bib_file = malpedia_bib.bib_library

    bib_parser = ParseMalpediaBibFile(base_path, bib_file)

    # bib_parser.save_dict_as_json()

    # scraper = await Scraper.create(url="klk", output_path="jvyg")

    # results = await scraper.scrape_images()

    # await scraper.save_metadata(Path("iweyugbdiwebd"))


if __name__ == "__main__":
    asyncio.run(main())
