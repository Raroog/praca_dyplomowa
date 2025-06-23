import json
import re
from pathlib import Path
from typing import Dict, List


def clean_metadata_filenames(directory: Path, dry_run: bool = True) -> int | None:
    """
    Clean filenames in clean_metadata.json by removing extra suffixes.

    Args:
        directory: Directory containing clean_metadata.json
        dry_run: If True, only show what would be changed

    Returns:
        Number of filename entries that were cleaned
    """
    if not directory.exists() or not directory.is_dir():
        print(f"Directory {directory} does not exist or is not a directory")
        return None

    metadata_file = directory / "clean_metadata.json"
    if not metadata_file.exists():
        print(f"clean_metadata.json not found in {directory}")
        return None

    # Read the metadata
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)

    if not isinstance(metadata_list, list):
        raise ValueError(f"Expected list in {metadata_file}, got {type(metadata_list)}")

    print(f"Processing: {directory}")

    # Process each entry in metadata
    cleaned_count = 0
    updated_metadata = []

    for entry in metadata_list:
        if not isinstance(entry, dict):
            updated_metadata.append(entry)
            continue

        # Copy the entry
        updated_entry = entry.copy()

        if "filename" in entry:
            original_filename = entry["filename"]
            cleaned_filename = clean_filename(original_filename)

            if cleaned_filename != original_filename:
                updated_entry["filename"] = cleaned_filename
                cleaned_count += 1
                print(f"  Cleaned: {original_filename} -> {cleaned_filename}")
            else:
                print(f"  OK: {original_filename}")

        updated_metadata.append(updated_entry)

    # Create new metadata file with prefix "_"
    new_metadata_file = directory / "_clean_metadata.json"

    if dry_run:
        print(f"  Would create: {new_metadata_file.name}")
        print(f"  Would clean {cleaned_count} filename entries")
    else:
        with open(new_metadata_file, "w", encoding="utf-8") as f:
            json.dump(updated_metadata, f, indent=2, ensure_ascii=False)
        print(f"  Created: {new_metadata_file.name}")
        print(f"  Cleaned {cleaned_count} filename entries")

    return cleaned_count


def clean_filename(filename: str) -> str:
    """
    Clean a filename by removing extra suffixes after image extensions.

    Examples:
        "image.png!small" -> "image.png"
        "photo.jpg!large" -> "photo.jpg"
        "file.jpeg!thumb" -> "file.jpeg"
        "normal.png" -> "normal.png" (unchanged)

    Args:
        filename: Original filename that may have extra suffixes

    Returns:
        Cleaned filename with extra suffixes removed
    """
    # Pattern to match image extensions followed by extra content
    image_pattern = re.compile(r"(.+\.(png|jpg|jpeg|gif|webp))", re.IGNORECASE)

    match = image_pattern.match(filename)
    if match:
        return match.group(1)

    # If no image extension pattern found, return original
    return filename


# Usage example
if __name__ == "__main__":
    # Process a single directory
    images_dir = Path("/home/bartek/Kod/PD/praca_dyplomowa/dane/scraping")
    for scrap_path in images_dir.glob("*"):
        print("=== DRY RUN ===")
        clean_metadata_filenames(scrap_path, dry_run=False)
