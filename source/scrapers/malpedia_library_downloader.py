import json
import logging
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

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
    blacklist: list = ["youtube.com", "twitter.com"]

    def __init__(self, path: str, bib_library: Library):
        self.path = Path(path)
        self.bib_library = bib_library
        self.bib_library_entries = self.bib_library.entries
        self.bib_list_of_dicts = [
            self.bib_entry_fields_to_url_title_dict(entry)
            for entry in self.bib_library_entries
        ]

    def bib_entry_fields_to_url_title_dict(self, entry):
        result = {
            field.key: field.value
            for field in entry.fields
            if field.key in ["urldate", "url", "title", "language"]
        }
        logger.info("Parsed bib entry fields to url-title dict")
        return result

    def bib_lib_entries_titles_links(self):
        result = [
            self.bib_entry_fields_to_url_title_dict(library_entry.fields)
            for library_entry in self.bib_library_entries
            if urlparse(
                self.bib_entry_fields_to_url_title_dict(library_entry.fields)[
                    "url"
                ].replace("www.", "")
            ).hostname
            not in self.blacklist
        ]
        logger.info("Parsed bib file to list of url-title dicts")
        return result

    def save_dict_as_json(self):
        current_date = datetime.now().strftime("%Y-%m-%d")
        path_to_save = self.path / current_date / "bibs.json"
        with open(path_to_save, "w") as file:
            json.dump(self.bib_list_of_dicts, file, indent=4)
        logger.info("Saved list of url-title dicts as json at: %s", path_to_save)
