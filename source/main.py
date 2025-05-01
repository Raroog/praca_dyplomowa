import asyncio
import logging
from pathlib import Path
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


async def process_site(session, site_data_dict, base_path, blacklist):
    """Process a single site with error handling"""
    url = site_data_dict["url"]
    if urlparse(url.replace("www.", "")).hostname in blacklist:
        logger.info(f"Skipping blacklisted site: {url}")
        return {"url": url, "status": "skipped_blacklist"}

    try:
        title_path = (
            site_data_dict["title"]
            .strip("{}")
            .strip(".")
            .replace(" ", "_")
            .replace("/", "_")
        )
        output_path = f"{base_path}/{title_path}"

        if Path(output_path).exists():
            logger.info(f"Skipping existing directory: {output_path} for URL: {url}")
            return {"url": url, "status": "skipped_existing"}

        scraper = await Scraper.create(
            url=url,
            output_path=output_path,
            session=session,
            timeout=30,  # 30 second timeout for requests
        )

        if not scraper.requests_html:
            return {
                "url": url,
                "status": "failed_download",
                "code": scraper.status_code,
            }

        results = await scraper.scrape_images()

        if results["successes"]:
            metadata_path = f"{output_path}/metadata.json"
            text_w_image_markers_path = f"{output_path}/text_w_image_markers.txt"
            clean_text_path = f"{output_path}/clean_text.txt"

            await scraper.save_metadata(metadata_path)
            await scraper.save_text_w_images(text_w_image_markers_path)
            await scraper.save_clean_text(clean_text_path)

            return {
                "url": url,
                "status": "success",
                "images": f"{results['successes']}/{results['total']}",
            }
        else:
            return {"url": url, "status": "no_images"}

    except Exception as e:
        logger.error(f"Error processing {url}: {str(e)}")
        return {"url": url, "status": "error", "message": str(e)}


async def main():
    config = load_config()

    setup_logging(config)
    base_path = config.get("scraping", {}).get("base_path")

    # Get the number of concurrent connections from config or default to 50
    # With 14th gen Intel, 32GB RAM and 900 Mbps download, we can handle higher concurrency
    max_concurrent = config.get("scraping", {}).get("max_concurrent", 50)

    malpedia_bib = GetMalpediaBibFile()
    bib_file = malpedia_bib.bib_library

    bib_parser = ParseMalpediaBibFile(base_path, bib_file)
    bib_parser.save_dict_as_json()

    # Create a semaphore to limit concurrent connections
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create a persistent aiohttp session with optimized connection pooling
    conn = aiohttp.TCPConnector(
        limit=max_concurrent,
        ttl_dns_cache=600,  # Longer DNS cache for better performance
        limit_per_host=10,  # Limit connections per host to avoid overwhelming servers
        force_close=False,  # Keep connections alive
        enable_cleanup_closed=True,  # Clean up closed connections
    )
    # Increase default timeout for session but keep individual request timeouts
    timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_read=30)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:

        async def process_with_semaphore(site_data):
            async with semaphore:
                return await process_site(
                    session, site_data, base_path, bib_parser.blacklist
                )

        # Process all sites concurrently but limited by the semaphore

        tasks = [process_with_semaphore(site) for site in bib_parser.bib_list_of_dicts]

        print("tasks: ", len(tasks))

        # Process sites in batches to avoid overwhelming memory
        # With 32GB RAM, we can handle larger batches
        batch_size = 200
        total_sites = len(tasks)

        for i in range(0, total_sites, batch_size):
            batch = tasks[i : i + batch_size]
            results = await asyncio.gather(*batch)

            # Log batch results
            successes = sum(1 for r in results if r.get("status") == "success")
            failures = sum(
                1
                for r in results
                if r.get("status") not in ["success", "skipped_blacklist"]
            )

            logger.info(
                f"Batch {i // batch_size + 1} completed: {successes} successful, {failures} failed"
            )

            # Print progress
            processed = min(i + batch_size, total_sites)
            print(
                f"Progress: {processed}/{total_sites} sites processed ({processed / total_sites * 100:.1f}%)"
            )


if __name__ == "__main__":
    # Import here to avoid circular imports
    import aiohttp

    asyncio.run(main())
