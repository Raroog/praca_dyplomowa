import json
from pathlib import Path

# id to lokalizacja obrazu w tekście

project_path = Path(__file__).parents[1]
ocr_results = project_path / "dane/ocr_results.json"
scraping_results = project_path / "dane/scraping"

with open(ocr_results, "r") as file:
    data = json.load(file)

scraping_results_metadata_paths = scraping_results.glob("**/*_clean_metadata.json")
for metadata_path in list(scraping_results_metadata_paths)[:3]:
    id_text = {}
    dirname = metadata_path.parts[-2]
    print(dirname)
    print(data.get(dirname))
    ocr_data = data.get(dirname)
    if ocr_data is None:
        continue
    ocr_data_filenames = [elem["filename"] for elem in ocr_data]
    with open(metadata_path, "r") as file:
        metadata = json.load(file)
    reduced_metadata = {
        elem["filename"]: elem["id"]
        for elem in metadata
        if elem["filename"] in ocr_data_filenames
    }
    print(reduced_metadata)
    final_dir_data = {
        reduced_metadata[elem["filename"]]: elem["text"] for elem in ocr_data
    }
    print(final_dir_data)
