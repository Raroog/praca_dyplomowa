import json
import re
from difflib import ndiff
from pathlib import Path


class Assemble_Clean_Text:
    IMG_MARKER_PATTERN = r"\+\s\[IMG:\d{1,2}\]"

    def __init__(self, trafilatura_text: str, img_markers_text: str, url: str) -> None:
        self.trafilatura_text = trafilatura_text
        self.trafilatura_words_list = self.split_ttext_to_words()
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
        return words_list

    def assemble_clean_text_w_img_markers(self):
        clean_words_list = []
        for word in self.words_list_diff:
            if not word.startswith("+") or re.fullmatch(self.IMG_MARKER_PATTERN, word):
                clean_words_list.append(word)
        return clean_words_list


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

    def read_text_from_image(self):
        pass
