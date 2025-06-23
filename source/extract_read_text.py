import json
from pathlib import Path

path = Path("/home/bartek/Kod/PD/praca_dyplomowa/dane/ocr_results")
paths = list(path.glob("*.json"))

final_json = {}
for json_path in paths:
    # print(json_path)
    with open(json_path, "r") as file:
        data = json.load(file)
    for response in data["responses"]:
        text = response.get("fullTextAnnotation", {}).get("text")
        if not text:
            continue
        uri = response.get("context", {}).get("uri")
        filename = Path(uri).name
        dirname = Path(uri).parts[-2]
        if dirname not in final_json:
            final_json.update({dirname: [{"filename": filename, "text": text}]})
        else:
            final_json[dirname].append({"filename": filename, "text": text})
        # print(f"uri: {uri}")
        # print(f"filename: {filename}")
        # print(f"dirname: {dirname}")
# print(final_json)
file_path = Path("/home/bartek/Kod/PD/praca_dyplomowa/dane/ocr_results.json")
with file_path.open("w") as f:
    json.dump(final_json, f, indent=2)
