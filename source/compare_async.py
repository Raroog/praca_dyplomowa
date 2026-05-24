import asyncio
import logging
import shutil
import time
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
import yaml
from logging_setup import setup_logging
from scrapers.malpedia_library_downloader import (
    GetMalpediaBibFile,
    ParseMalpediaBibFile,
)
from scrapers.scraper import Scraper

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except Exception:
        return {}


async def process_site(
    session: aiohttp.ClientSession,
    site_data_dict: dict,
    base_path: Path,
    blacklist: list,
) -> dict:
    url = site_data_dict["url"]

    hostname = urlparse(url.replace("www.", "")).hostname
    if hostname in blacklist:
        return {"url": url, "status": "skipped_blacklist", "size_bytes": 0}

    try:
        title_path = (
            site_data_dict["title"]
            .strip("{}")
            .strip(".")
            .replace(" ", "_")
            .replace("/", "_")
        )

        output_dir = base_path / title_path
        output_dir.mkdir(parents=True, exist_ok=True)

        scraper = await Scraper.create(
            url=url,
            output_path=str(output_dir),
            session=session,
            timeout=30,
        )

        if not scraper.requests_html:
            return {
                "url": url,
                "status": "failed_download",
                "code": scraper.status_code,
                "size_bytes": 0,
            }

        # --- PEŁNE POBIERANIE (HTML + OBRAZY) ---
        results = await scraper.scrape_images()

        if results["successes"] or scraper.requests_html:
            # Zapisujemy metadane i tekst (żeby procedura była pełna)
            await scraper.save_metadata(str(output_dir / "metadata.json"))
            await scraper.save_text_w_images(
                str(output_dir / "text_w_image_markers.txt")
            )
            await scraper.save_clean_text(str(output_dir / "clean_text.txt"))

            # Zliczamy całkowity rozmiar
            total_bytes = sum(
                f.stat().st_size for f in output_dir.rglob("*") if f.is_file()
            )

            return {
                "url": url,
                "status": "success",
                "images": f"{results['successes']}/{results['total']}",
                "size_bytes": total_bytes,
            }
        else:
            return {"url": url, "status": "no_content", "size_bytes": 0}

    except Exception as e:
        return {"url": url, "status": "error", "message": str(e), "size_bytes": 0}


async def main():
    print("--- PRZYGOTOWANIE DANYCH (ASYNC FULL) ---")
    config = load_config()
    setup_logging(config)

    # Czyszczenie katalogu
    benchmark_base_path = Path("benchmark_data_async")
    if benchmark_base_path.exists():
        shutil.rmtree(benchmark_base_path)
    benchmark_base_path.mkdir(parents=True, exist_ok=True)

    # Pobieranie listy
    malpedia_bib = GetMalpediaBibFile()
    bib_parser = ParseMalpediaBibFile(
        str(benchmark_base_path), malpedia_bib.bib_library
    )

    # Filtrowanie listy (żeby była identyczna z Sync)
    all_clean_urls = []
    for entry in bib_parser.bib_list_of_dicts:
        url = entry["url"]
        hostname = urlparse(url.replace("www.", "")).hostname
        if hostname not in bib_parser.blacklist:
            all_clean_urls.append(entry)

    # WYBÓR 100 LINKÓW
    TARGET_COUNT = 100
    tasks_source = (
        all_clean_urls[:TARGET_COUNT]
        if len(all_clean_urls) >= TARGET_COUNT
        else all_clean_urls
    )

    print(f"Liczba zadań do wykonania: {len(tasks_source)}")

    max_concurrent = config.get("scraping", {}).get("max_concurrent", 50)
    semaphore = asyncio.Semaphore(max_concurrent)

    conn = aiohttp.TCPConnector(limit=max_concurrent, ttl_dns_cache=600)
    timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_read=30)

    print("\n--- START POMIARU CZASU (ASYNC) ---")
    start_time = time.perf_counter()

    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:

        async def process_with_semaphore(site_data):
            async with semaphore:
                return await process_site(
                    session, site_data, benchmark_base_path, bib_parser.blacklist
                )

        tasks = [process_with_semaphore(site) for site in tasks_source]
        results = await asyncio.gather(*tasks)

    end_time = time.perf_counter()
    total_duration = end_time - start_time

    # Podsumowanie
    success_count = sum(1 for r in results if r.get("status") == "success")
    total_mb = sum(r.get("size_bytes", 0) for r in results) / (1024 * 1024)

    print("\n" + "=" * 40)
    print(f"WYNIKI BENCHMARKU (ASYNCIO - FULL)")
    print("=" * 40)
    print(f"Przetworzono URLi: {len(results)}")
    print(f"Pobrano poprawnie: {success_count}")
    print(f"Całkowity czas: {total_duration:.2f} s")
    print(f"Rozmiar danych: {total_mb:.2f} MB")
    print("=" * 40)


if __name__ == "__main__":
    asyncio.run(main())
