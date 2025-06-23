import json
from pathlib import Path

from google.cloud import storage

client = storage.Client()

bucket = client.bucket("text-detection")

blobs = bucket.list_blobs()
blobs_list = list(blobs)

for blob in blobs_list:
    results = blob.download_as_text()
    results_json = json.loads(results)
    with open(
        f"/home/bartek/Kod/PD/praca_dyplomowa/dane/ocr_results/{blob.name}",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
