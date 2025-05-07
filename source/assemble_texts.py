import json
import logging
import re
from difflib import ndiff
from pathlib import Path

import yaml
from logging_setup import setup_logging

logger = logging.getLogger(__name__)


def load_config(config_path="/home/bartek/Kod/PD/praca_dyplomowa/config.yaml"):
    """Load configuration from YAML file"""
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


class Assemble_Clean_Text:
    IMG_MARKER_PATTERN = r"\[IMG:(\d+)\]"

    def __init__(
        self, trafilatura_text: str, img_markers_text: str, path: Path
    ) -> None:
        self.path = path
        self.trafilatura_text = trafilatura_text
        self.trafilatura_words_list = ["<!*start*!>"]
        self.trafilatura_words_list.extend(self.split_ttext_to_words())
        # self.first_trafilatura_text_word = self.trafilatura_words_list[0].strip("- ")
        self.first_trafilatura_text_word = "<!*start*!>"
        self.img_markers_text = img_markers_text
        self.img_markers_words_list = self.img_markers_text.split(" ")
        logger.debug("Split text with image markers into list for %s", self.path.stem)
        self.words_list_diff = list(
            ndiff(
                self.trafilatura_words_list,
                self.img_markers_words_list,
            )
        )
        self.reversed_words_list_diff = list(
            ndiff(
                list(reversed(self.trafilatura_words_list)),
                list(reversed(self.img_markers_words_list)),
            )
        )
        self.clean_text_w_img_markers = self.assemble_clean_text_w_img_markers()
        self.image_markers_from_clean_text = (
            self.extract_image_markers_from_clean_text()
        )
        self.clean_metadata = self.filter_metada_by_image_markers()

    def split_ttext_to_words(self):
        words_list = []
        for line in self.trafilatura_text.splitlines():
            words_list.extend(line.split(" "))
        words_list.append("<!*stop*!>")
        logger.debug("Split trafiltura text into list of words for %s", self.path.stem)
        return words_list

    def assemble_clean_text_w_img_markers(self):
        clean_words_list = []
        for word in self.words_list_diff:
            if not word.startswith("+") or re.search(self.IMG_MARKER_PATTERN, word):
                clean_words_list.append(word.strip("-+? "))
        start_index = clean_words_list.index(self.first_trafilatura_text_word)
        stop_index = clean_words_list.index("<!*stop*!>")
        logger.info(
            "Assembled trafilatura text with image markers text for %s", self.path.stem
        )
        return " ".join(clean_words_list[start_index:stop_index])

    def extract_image_markers_from_clean_text(self):
        logger.debug("Extracting image markers from clean text")
        return [
            int(match)
            for match in re.findall(
                self.IMG_MARKER_PATTERN, self.clean_text_w_img_markers
            )
        ]

    def filter_metada_by_image_markers(self):
        json_file_path = self.path / "metadata.json"
        with open(json_file_path, "r") as file:
            metadata = json.load(file)
        logger.debug("Filtering metadata by clean image markers")
        return [
            element
            for element in metadata
            if int(element["id"]) in self.image_markers_from_clean_text
        ]

    def save_assembled_text(self):
        save_path = f"{self.path}/clean_text_w_image_markers.txt"
        with open(save_path, "w") as file:
            file.write(self.clean_text_w_img_markers)
        logger.info("Saved assembled text at %s", save_path)

    def save_clean_metada(self):
        clean_json_path = self.path / "clean_metadata.json"
        with open(clean_json_path, "w") as file:
            json.dump(self.clean_metadata, file)
        logger.info("Saved clean metadata at %s", clean_json_path)


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
    config = load_config()

    setup_logging(config)
    scraped_path = Path("/home/bartek/Kod/PD/praca_dyplomowa/dane/scraping")
    for site_path in list(scraped_path.glob("*"))[14:18]:
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
        print(site_path.stem)
        text_cleaner = Assemble_Clean_Text(ttext, image_text, site_path)
        # print("<>" * 40)
        print(text_cleaner.trafilatura_text)
        print("--" * 40)
        print(text_cleaner.words_list_diff)
        print("--" * 40)
        print(text_cleaner.reversed_words_list_diff)
        # print(text_cleaner.clean_text_w_img_markers)
        print("--" * 40)
        # print(text_cleaner.image_markers_from_clean_text)
        # print("--" * 40)
        # print(text_cleaner.first_trafilatura_text_word)

        # text_cleaner.save_assembled_text()
        # text_cleaner.save_clean_metada()
        print("<>" * 40)
