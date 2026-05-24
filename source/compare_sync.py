import logging
import shutil
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
import yaml
from bs4 import BeautifulSoup
from logging_setup import setup_logging
from scrapers.malpedia_library_downloader import (
    GetMalpediaBibFile,
    ParseMalpediaBibFile,
)

logger = logging.getLogger(__name__)

# Limit obrazów zgodny z Twoim scraperem
MAX_IMAGES = 20


def load_config(config_path: str = "config.yaml") -> dict:
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except Exception:
        return {}


def download_image(url, path):
    try:
        r = requests.get(url, stream=True, timeout=10)
        if r.status_code == 200:
            with open(path, "wb") as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
            return True
    except Exception:
        return False
    return False


def sync_process_site(url: str, title: str, base_path: Path, blacklist: list) -> dict:
    hostname = urlparse(url.replace("www.", "")).hostname
    if hostname in blacklist:
        return {"url": url, "status": "skipped", "size_bytes": 0}

    title_clean = title.strip("{}").strip(".").replace(" ", "_").replace("/", "_")
    output_dir = base_path / title_clean
    output_dir.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0"
    }

    try:
        # 1. HTML
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            return {"url": url, "status": "failed", "size_bytes": 0}

        with open(output_dir / "source.html", "wb") as f:
            f.write(resp.content)

        # 2. Parsowanie i Obrazy
        soup = BeautifulSoup(resp.content, "html.parser")
        images = soup.find_all("img")

        count = 0
        for img in images[:MAX_IMAGES]:
            src = img.get("src")
            if not src:
                continue

            img_url = urljoin(url, src)
            img_name = Path(urlparse(img_url).path).name
            if not img_name:
                img_name = f"img_{count}.jpg"

            if download_image(img_url, output_dir / img_name):
                count += 1

        total_bytes = sum(
            f.stat().st_size for f in output_dir.rglob("*") if f.is_file()
        )

        return {
            "url": url,
            "status": "success",
            "images": count,
            "size_bytes": total_bytes,
        }

    except Exception as e:
        return {"url": url, "status": "error", "size_bytes": 0}


def main():
    print("--- PRZYGOTOWANIE DANYCH (SYNC FULL) ---")
    config = load_config()
    setup_logging(config)

    # Czyszczenie katalogu
    benchmark_base_path = Path("benchmark_data_sync")
    if benchmark_base_path.exists():
        shutil.rmtree(benchmark_base_path)
    benchmark_base_path.mkdir(parents=True, exist_ok=True)

    # Pobieranie listy (Identycznie jak w Async)
    malpedia_bib = GetMalpediaBibFile()
    bib_parser = ParseMalpediaBibFile(
        str(benchmark_base_path), malpedia_bib.bib_library
    )

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
    print("\n--- START POMIARU CZASU (SYNC) ---")

    start_time = time.perf_counter()

    results = []
    for i, item in enumerate(tasks_source):
        if i % 5 == 0:
            print(f"Postęp: {i}/{len(tasks_source)}...")
        res = sync_process_site(
            item["url"], item["title"], benchmark_base_path, bib_parser.blacklist
        )
        results.append(res)

    end_time = time.perf_counter()
    total_duration = end_time - start_time

    success_count = sum(1 for r in results if r.get("status") == "success")
    total_mb = sum(r.get("size_bytes", 0) for r in results) / (1024 * 1024)

    print("\n" + "=" * 40)
    print(f"WYNIKI BENCHMARKU (SYNC - FULL)")
    print("=" * 40)
    print(f"Przetworzono URLi: {len(results)}")
    print(f"Pobrano poprawnie: {success_count}")
    print(f"Całkowity czas: {total_duration:.2f} s")
    print(f"Rozmiar danych: {total_mb:.2f} MB")
    print("=" * 40)


if __name__ == "__main__":
    main()
