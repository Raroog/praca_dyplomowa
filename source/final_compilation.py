import json
import logging
from pathlib import Path

# id to lokalizacja obrazu w tekście
logger = logging.getLogger(__name__)


project_path = Path(__file__).parents[1]
ocr_results = project_path / "dane/ocr_results.json"
scraping_results = project_path / "dane/scraping"

with open(ocr_results, "r") as file:
    data = json.load(file)

scraping_results_metadata_paths = scraping_results.glob("**/*_clean_metadata.json")
scraping_path = Path("/home/bartek/Kod/PD/praca_dyplomowa/dane/scraping")
success = 0
no_ocr = 0
dir_not_found = 0
for metadata_path in scraping_results_metadata_paths:
    id_text = {}
    dirname = metadata_path.parts[-2]
    # print(data.get(dirname))
    ocr_data = data.get(dirname)
    if ocr_data is None:
        logger.error(f"{dirname}: No OCR data")
        no_ocr += 1
        continue
    ocr_data_filenames = [elem["filename"] for elem in ocr_data]
    with open(metadata_path, "r") as file:
        metadata = json.load(file)
    reduced_metadata = {
        elem["filename"]: elem["id"]
        for elem in metadata
        if elem["filename"] in ocr_data_filenames
    }
    final_dir_data = {
        reduced_metadata[elem["filename"]]: elem["text"] for elem in ocr_data
    }
    # print(final_dir_data)
    json_path = scraping_path / dirname / "img_id_ocr_text.json"
    try:
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(final_dir_data, f, indent=2, ensure_ascii=False)
        logger.info(f"{dirname}: saved successfully!")
        success += 1
    except FileNotFoundError:
        logger.error(f"{dirname} does not exist! {80 * '<>'}")
        dir_not_found += 1
total = success + no_ocr + dir_not_found
print(f"success: {(success / total) * 100}%; n: {success}")
print(f"no_ocr: {(no_ocr / total) * 100}%; n: {no_ocr}")
print(f"dir_not_found: {(dir_not_found / total) * 100}%; n: {dir_not_found}")
