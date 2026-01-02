from pathlib import Path

cleaned = Path("/home/bartek/Kod/PD/praca_dyplomowa/dane/texts/cleaned_texts")

ocr = Path("/home/bartek/Kod/PD/praca_dyplomowa/dane/texts/ocr_enriched_texts")

ocr_texts = list(ocr.glob("**/clean_text*.txt"))

cleaned_texts = list(cleaned.glob("**/LLM_clean_text.txt"))

cleaned_texts = list(cleaned.glob("*"))


print(len(cleaned_texts))
