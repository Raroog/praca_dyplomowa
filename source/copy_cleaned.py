import re
import shutil
from pathlib import Path
from typing import List, Optional


def copy_clean_images_in_place(
    directory: Path,
    dry_run: bool = True,
) -> List[tuple[Path, Path]] | None:
    """
    Copy files with cleaned names to the same directory with a suffix.

    Args:
        directory: Directory containing the files
        suffix: Suffix to add to cleaned files (default: "_clean")
        dry_run: If True, only show what would be copied
    """
    if not directory.exists() or not directory.is_dir():
        return None
    image_pattern = re.compile(r"(.+)\.(png|jpg|jpeg)", re.IGNORECASE)
    copied_files = []

    for file_path in directory.iterdir():
        if not file_path.is_file():
            continue

        # Check if this looks like a mangled image filename
        original_match = image_pattern.match(file_path.name)
        if original_match:
            base_name = original_match.group(1)
            extension = original_match.group(2)

            # Only process if the filename has extra content after the extension
            if len(file_path.name) > len(f"{base_name}.{extension}"):
                clean_name = f"{base_name}.{extension}"
                target_path = directory / clean_name

                if target_path.exists():
                    print(f"Warning: {target_path} already exists, skipping")
                    continue

                copied_files.append((file_path, target_path))

                if dry_run:
                    print(f"Would copy: {file_path.name} -> {clean_name}")
                else:
                    try:
                        shutil.copy2(file_path, target_path)
                        print(f"Copied: {file_path.name} -> {clean_name}")
                    except Exception as e:
                        print(f"Error copying {file_path.name}: {e}")

    return copied_files


# Usage examples
if __name__ == "__main__":
    # Set your directory path
    images_dir = Path("/home/bartek/Kod/PD/praca_dyplomowa/dane/scraping")
    for scrap_path in images_dir.glob("*"):
        # Option 2: Copy to same directory with suffix
        print("\n=== DRY RUN: Copy with suffix in same directory ===")
        copy_clean_images_in_place(scrap_path, dry_run=False)
