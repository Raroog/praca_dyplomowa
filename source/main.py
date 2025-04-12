# from .malpedia_library_downloader import GetMalpediaBibFile, ParseMalpediaBibFile
import asyncio
from pathlib import Path

from .scrapers.scraper import Scraper


async def main():
    scraper = await Scraper.create(url="klk", output_path="jvyg")

    results = await scraper.scrape_images()

    await scraper.save_metadata(Path("iweyugbdiwebd"))


if __name__ == "__ main __":
    asyncio.run(main())
