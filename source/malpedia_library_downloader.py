from urllib.parse import urlparse

import bibtexparser
import requests
from bibtexparser import Library


class GetMalpediaBibFile:
    URL: str = "https://malpedia.caad.fkie.fraunhofer.de/library/download"

    def __init__(self):
        self.response = requests.get(self.URL)
        self.status_code = self.response.status_code
        self.bib_txt = self.response.text if self.status_code == 200 else None
        if self.bib_txt is not None:
            self.bib_library = bibtexparser.parse_string(self.bib_txt)


class ParseMalpediaBibFile:
    blacklist: list = ["youtube.com", "twitter.com"]

    def __init__(self, bib_library: Library):
        self.bib_library = bib_library
        self.bib_library_entries = self.bib_library.entries
        self.bib_list_of_dicts = [
            self.bib_entry_fields_to_url_title_dict(entry)
            for entry in self.bib_library_entries
        ]

    def bib_entry_fields_to_url_title_dict(self, entry):
        return {
            field.key: field.value
            for field in entry.fields
            if field.key in ["url", "title"]
        }

    def bib_lib_entries_titles_links(self):
        return [
            self.bib_entry_fields_to_url_title_dict(library_entry.fields)
            for library_entry in self.bib_library_entries
            if urlparse(
                self.bib_entry_fields_to_url_title_dict(library_entry.fields)[
                    "url"
                ].replace("www.", "")
            ).hostname
            not in self.blacklist
        ]
