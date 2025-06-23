import json
from pathlib import Path

from google.cloud import storage

client = storage.Client()

bucket = client.bucket("scraped_images")

blobs = bucket.list_blobs(prefix="scraping")
blobs_list = list(blobs)

gcs_data = []

for blob in blobs_list:
    blob_path = Path(blob.name)
    dir_path = blob_path.parent
    if blob_path.name == "_clean_metadata.json":
        metadata = blob.download_as_text()
        metadata_list = json.loads(metadata)
        images = [
            f"gs://scraped_images/{dir_path / file['filename']}"
            for file in metadata_list
            if Path(file["filename"]).suffix in [".jpg", ".jpeg", "png"]
        ]
        gcs_data.append({"name": dir_path.name, "images": images})

with open(
    "/home/bartek/Kod/PD/praca_dyplomowa/dane/images_preocr.json", "w", encoding="utf-8"
) as f:
    json.dump(gcs_data, f, indent=2, ensure_ascii=False)
