import json
import re
from difflib import ndiff
from pathlib import Path


class Assemble_Clean_Text:
    IMG_MARKER_PATTERN = r"\+\s\[IMG:\d{1,2}\](?:\w*)"

    def __init__(self, trafilatura_text: str, img_markers_text: str) -> None:
        self.trafilatura_text = trafilatura_text
        self.trafilatura_words_list = self.split_ttext_to_words()
        self.first_trafilatura_text_word = self.trafilatura_words_list[0]
        self.img_markers_text = img_markers_text
        self.img_markers_words_list = self.img_markers_text.split(" ")
        self.words_list_diff = list(
            ndiff(self.trafilatura_words_list, self.img_markers_words_list)
        )
        self.clean_text_w_img_markers = self.assemble_clean_text_w_img_markers()

    def split_ttext_to_words(self):
        words_list = []
        for line in self.trafilatura_text.splitlines():
            words_list.extend(line.split(" "))
        words_list.append("stop")
        return words_list

    def assemble_clean_text_w_img_markers(self):
        clean_words_list = []
        for word in self.words_list_diff:
            if not word.startswith("+") or re.fullmatch(self.IMG_MARKER_PATTERN, word):
                clean_words_list.append(word.strip())
        start_index = clean_words_list.index(self.first_trafilatura_text_word)
        stop_index = clean_words_list.index("- stop") + 1
        return clean_words_list[start_index:stop_index]


class Assemble_Final_Text:
    def __init__(
        self, clean_text_w_img_markers: list[str], image_metada_path: str
    ) -> None:
        self.clean_text_w_img_markers = clean_text_w_img_markers
        self.image_metada_path = image_metada_path
        self.image_metada = self.read_image_data()

    def read_image_data(self):
        with open(self.image_metada_path, "r", encoding="utf-8") as file:
            image_metada = json.load(file)
        return image_metada

    def list_candidates_for_OCR(self):
        pass

    def read_text_from_image(self):
        pass


if __name__ == "__main__":
    scraped_path = Path("/home/bartek/Kod/PD/praca_dyplomowa/dane/scraping")
    for site_path in list(scraped_path.glob("*"))[:10]:
        if str(site_path).endswith("json"):
            continue
        if not list(site_path.glob("*")):
            continue
        ttext_path = site_path / "clean_text.txt"
        image_text_path = site_path / "text_w_image_markers.txt"
        with open(ttext_path, "r") as file:
            ttext = file.read()
        with open(image_text_path, "r") as file:
            image_text = file.read()
        text_cleaner = Assemble_Clean_Text(ttext, image_text)
        # print("<>" * 40)
        # print(site_path.stem)
        # print(text_cleaner.trafilatura_words_list)
        # print("--" * 40)
        # print(text_cleaner.img_markers_words_list)
        # print("--" * 40)
        # print(text_cleaner.words_list_diff)
        # print("--" * 40)
        print(text_cleaner.clean_text_w_img_markers)
        # print("<>" * 40)
