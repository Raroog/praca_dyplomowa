import json
import re
from pathlib import Path


def insert_ocr(ocr_data, text):
    for i, ocr_data in enumerate(tuple(ocr_data.items())):
        n, ocr_text = ocr_data
        ocr_text_modf = f"\n<OCR text #{i + 1}>\n{ocr_text}\n</OCR text #{i + 1}>\n"
        pattern = rf"\[IMG\:{n}\]"
        text = re.sub(pattern, lambda m: ocr_text_modf, text)
    return text


def remove_img_markers(text):
    pattern = r"\[IMG\:\d{1,}\]"
    clean_text = re.sub(pattern, "", text)
    return clean_text


def save_multiline_text(filepath: Path, text: str) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(text, encoding="utf-8")


ocr_nones = 0
clean_text_nones = 0
base_data_path = Path("/home/bartek/Kod/PD/praca_dyplomowa/dane/")
data_paths = list(base_data_path.glob("scraping/*"))
for data_path in data_paths:
    # loading ocr data
    ocr_data = None
    ocr_data_path = data_path / "img_id_ocr_text.json"
    try:
        with ocr_data_path.open("r", encoding="utf-8") as file:
            ocr_data = json.load(file)
    except FileNotFoundError:
        ocr_nones += 1

    # loading cleaned scraped text
    clean_text = None
    clean_text_path = data_path / "clean_text_w_image_markers.txt"
    try:
        with clean_text_path.open("r", encoding="utf-8") as f:
            clean_text = f.read()
    except FileNotFoundError:
        clean_text_nones += 1
        continue
    final_text_path_root = base_data_path / "texts/ocr_enriched_texts" / data_path.stem
    if not ocr_data:
        final_text = remove_img_markers(clean_text)
        final_text_path = final_text_path_root / "clean_text.txt"
    else:
        clean_text_w_ocr = insert_ocr(ocr_data, clean_text)
        final_text = remove_img_markers(clean_text_w_ocr)
        final_text_path = final_text_path_root / "clean_text_w_ocr.txt"

    # print(final_text_path)
    save_multiline_text(final_text_path, final_text)


print(f"Failed ocr dirs: {ocr_nones / len(data_paths)}")
print(f"Failed clean text dirs: {clean_text_nones / len(data_paths)}")
