import json
import time

from batch_ocr import sample_async_batch_annotate_images

if __name__ == "__main__":
    total_start_time = time.perf_counter()
    path = "/home/bartek/Kod/PD/praca_dyplomowa/dane/images_preocr.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    elements = [
        item for sublist in [element["images"] for element in data] for item in sublist
    ]
    stop = len(elements)
    print(stop)
    for i in range(0, stop, 2000):
        start_time = time.perf_counter()
        batch = elements[i : i + 2000]
        prefix = f"{i} : {i + 2000}_"
        sample_async_batch_annotate_images(batch, prefix)
        elapsed = time.perf_counter() - start_time
        print(f"  ⏱️  Batch: {prefix}; Elapsed: {elapsed:.2f}s")
    total_elapsed = time.perf_counter() - total_start_time
    print(f"\n✅ Total processing time: {total_elapsed:.2f}s")
