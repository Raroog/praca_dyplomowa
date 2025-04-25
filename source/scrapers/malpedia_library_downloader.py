import json
import logging
import re
from datetime import datetime
from pathlib import Path

import bibtexparser
import requests
from bibtexparser import Library

logger = logging.getLogger(__name__)


class GetMalpediaBibFile:
    URL: str = "https://malpedia.caad.fkie.fraunhofer.de/library/download"

    def __init__(self):
        self.response = requests.get(self.URL)
        self.status_code = self.response.status_code
        self.bib_txt = self.response.text if self.status_code == 200 else None
        if self.bib_txt is not None:
            self.bib_library = bibtexparser.parse_string(self.bib_txt)
            logger.info("Created bib file")
        else:
            logger.warning("Could not download bib file")


class ParseMalpediaBibFile:
    # blacklist: list =

    def __init__(self, path: str, bib_library: Library):
        self.path = Path(path)
        self.bib_library = bib_library
        self.bib_library_entries = self.bib_library.entries
        self.bib_list_of_dicts = [
            self.bib_entry_fields_to_url_title_dict(entry)
            for entry in self.bib_library_entries
        ]
        self.blacklist = [
            "youtube.com",
            "twitter.com",
            "x.com",
            "tccontre.blogspot.com",
        ] + list(
            (
                self.extract_urls_from_log(
                    "/home/bartek/Kod/PD/praca_dyplomowa/logs/application.log"
                )
                | self.extract_urls_from_log(
                    "/home/bartek/Kod/PD/praca_dyplomowa/logs/application1.log"
                )
                | self.extract_urls_from_log(
                    "/home/bartek/Kod/PD/praca_dyplomowa/logs/application2.log"
                )
                | self.extract_urls_from_log(
                    "/home/bartek/Kod/PD/praca_dyplomowa/logs/application3.log"
                )
            )
        )

    def bib_entry_fields_to_url_title_dict(self, entry):
        result = {
            field.key: field.value
            for field in entry.fields
            if field.key in ["urldate", "url", "title", "language"]
        }
        # logger.info("Parsed bib entry fields to url-title dict")
        return result

    def extract_urls_from_log(self, log_file_path):
        urls = set()

        # Pattern to match "Started downloading data from url" followed by a URL
        pattern = r"Started downloading data from url\s+(https?://\S+)"

        with open(log_file_path, "r") as file:
            for line in file:
                match = re.search(pattern, line)
                if match:
                    url = match.group(1)
                    urls.add(url)

        return urls

    def save_dict_as_json(self):
        current_date = datetime.now().strftime("%Y-%m-%d")
        path_to_save = self.path / f"{current_date}_bibs.json"
        with open(path_to_save, "w") as file:
            json.dump(self.bib_list_of_dicts, file, indent=4)
        logger.info("Saved list of url-title dicts as json at: %s", path_to_save)
