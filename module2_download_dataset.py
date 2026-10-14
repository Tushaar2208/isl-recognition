"""
Module 2: Dataset Downloader and Organizer
==========================================

This script handles the downloading and organization of the INCLUDE-50 dataset.
It provides a multi-step workflow:
1. Metadata generation: Fetches metadata from HuggingFace, filtering to the 50 classes.
2. Video download: Downloads raw video files via Zenodo (default), Kaggle, or a local directory.
3. Verification: Ensures all files listed in the metadata are correctly downloaded.

Usage examples:
    # 1. Fetch metadata only
    python module2_download_dataset.py --metadata-only

    # 2. Download videos (choose one source)
    python module2_download_dataset.py --source zenodo
    python module2_download_dataset.py --source kaggle
    python module2_download_dataset.py --source local --local-path /path/to/videos

    # 3. Verify downloaded files
    python module2_download_dataset.py --verify
"""

import argparse
import csv
import json
import logging
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

# Import project configuration
from config import (
    ensure_dirs,
    METADATA_DIR,
    RAW_VIDEO_DIR,
    DATASET_INFO_CSV,
    CLASS_MAPPING_JSON,
    HF_DATASET_ID
)

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def fetch_metadata():
    """
    Fetch metadata from HuggingFace dataset, filter for INCLUDE-50, and save CSV/JSON files.
    """
    logging.info("Loading HuggingFace dataset metadata...")
    try:
        from datasets import load_dataset
    except ImportError:
        logging.error("Please install the 'datasets' package (pip install datasets).")
        return

    # Using 'streaming=True' might be problematic depending on HF script, but usually load_dataset works.
    try:
        dataset_dict = load_dataset(HF_DATASET_ID)
    except Exception as e:
        logging.error(f"Error loading dataset from HuggingFace: {e}")
        return

    metadata = []
    class_mapping = {}
    class_index_counter = 0

    for split_name, split_data in dataset_dict.items():
        for row in tqdm(split_data, desc=f"Processing {split_name} split"):
            # Filter by include_50 flag
            is_include_50 = row.get("include_50", False)
            if not is_include_50:
                continue

            # ── Extract fields using actual HF column names ─────────────
            # Columns are: parent_label, label, video_path, include_50
            video_path = row.get("video_path", "")          # e.g. "Adjectives/1. loud/MVI_9448.MOV"
            label = row.get("label", "Unknown")              # e.g. "1. loud"
            parent_label = row.get("parent_label", "")       # e.g. "Adjectives"

            # ── Signer ID ───────────────────────────────────────────────
            # The HF dataset does NOT include signer IDs directly.
            # Signer identity can be inferred later from the original
            # train.csv/test.csv if available, or from video properties.
            # For now, we assign "Unknown" and document this limitation.
            signer_id = "Unknown"

            video_filename = Path(video_path).name

            if label not in class_mapping:
                class_mapping[label] = class_index_counter
                class_index_counter += 1

            metadata.append({
                "video_filename": video_filename,
                "class_name": label,
                "class_index": class_mapping[label],
                "parent_category": parent_label,
                "signer_id": signer_id,
                "split": split_name,
                "video_path_in_archive": video_path
            })

    # Save to CSV
    ensure_dirs()
    
    with open(DATASET_INFO_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "video_filename", "class_name", "class_index", "parent_category",
            "signer_id", "split", "video_path_in_archive"
        ])
        writer.writeheader()
        writer.writerows(metadata)
        
    # Save class mapping
    with open(CLASS_MAPPING_JSON, "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=4)

    logging.info(f"Metadata saved. Total videos: {len(metadata)}")
    logging.info(f"Total classes found: {len(class_mapping)}")
    

def download_from_zenodo():
    """
    Download INCLUDE dataset parts from Zenodo, extract and filter INCLUDE-50 videos.
    """
    logging.info("Downloading from Zenodo API...")
    url = "https://zenodo.org/api/records/4010759"
    try:
        req = urllib.request.urlopen(url)
        res = json.loads(req.read())
    except Exception as e:
        logging.error(f"Failed to fetch Zenodo API: {e}")
        return

    files = res.get("files", [])
    if not files:
        logging.error("No files found in Zenodo record.")
        return

    # Load metadata to know which files to keep
    expected_classes = _get_expected_classes()
    if not expected_classes:
        logging.warning("No expected classes found. Run --metadata-only first.")
        return

    for f_info in files:
        download_url = f_info.get("links", {}).get("self")
        filename = f_info.get("key")
        if not download_url or not filename.endswith('.zip'):
            continue
            
        zip_path = RAW_VIDEO_DIR / filename
        _download_file(download_url, zip_path)
        
        # Extract and copy matching videos
        _extract_and_filter_zip(zip_path, expected_classes)


def download_from_kaggle():
    """
    Download INCLUDE dataset using Kaggle CLI.
    """
    logging.info("Downloading from Kaggle...")
    try:
        import subprocess
        subprocess.run(
            ["kaggle", "datasets", "download", "koushikdas/include-indian-sign-language", "-p", str(RAW_VIDEO_DIR)], 
            check=True
        )
    except Exception as e:
        logging.error(f"Failed to download from Kaggle. Ensure Kaggle CLI is installed and authenticated. Error: {e}")
        return

    expected_classes = _get_expected_classes()
    zip_path = RAW_VIDEO_DIR / "include-indian-sign-language.zip"
    if zip_path.exists():
        _extract_and_filter_zip(zip_path, expected_classes)
    else:
        logging.error("Kaggle zip file not found after download.")


def copy_from_local(local_path: str):
    """
    Copy videos from a local directory matching dataset_info.csv.
    """
    local_dir = Path(local_path)
    if not local_dir.exists() or not local_dir.is_dir():
        logging.error(f"Local path does not exist or is not a directory: {local_path}")
        return
        
    metadata = _load_metadata()
    if not metadata:
        logging.warning("No metadata found. Run --metadata-only first.")
        return

    expected_files = {row["video_filename"]: row["class_name"] for row in metadata}
    
    logging.info(f"Scanning local directory {local_path} for videos...")
    
    copied = 0
    for root, _, files in os.walk(local_dir):
        for file in files:
            if file.lower().endswith(('.mov', '.mp4')):
                if file in expected_files:
                    class_name = expected_files[file]
                    dest_dir = RAW_VIDEO_DIR / class_name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    
                    src_file = Path(root) / file
                    dest_file = dest_dir / file
                    if not dest_file.exists():
                        shutil.copy2(src_file, dest_file)
                        copied += 1
                        
    logging.info(f"Copied {copied} expected video files from local directory.")


def _download_file(url: str, dest: Path):
    """Download a file with a progress bar."""
    if dest.exists():
        logging.info(f"File {dest.name} already exists, skipping download.")
        return
        
    logging.info(f"Downloading {url} to {dest}...")
    try:
        with urllib.request.urlopen(url) as response, open(dest, 'wb') as out_file:
            total_size = int(response.info().get('Content-Length', 0))
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest.name) as pbar:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    pbar.update(len(chunk))
    except Exception as e:
        logging.error(f"Error downloading {url}: {e}")
        if dest.exists():
            dest.unlink()


def _get_expected_classes() -> set:
    """Read expected classes from metadata."""
    if not CLASS_MAPPING_JSON.exists():
        return set()
    with open(CLASS_MAPPING_JSON, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return set(mapping.keys())
    

def _load_metadata() -> list:
    """Load metadata from CSV."""
    if not DATASET_INFO_CSV.exists():
        return []
    metadata = []
    with open(DATASET_INFO_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata.append(row)
    return metadata


def _extract_and_filter_zip(zip_path: Path, expected_classes: set):
    """
    Extract zip, copy valid classes to RAW_VIDEO_DIR, and clean up.
    """
    logging.info(f"Extracting {zip_path.name}...")
    temp_dir = RAW_VIDEO_DIR / "temp_extract"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in tqdm(zip_ref.namelist(), desc="Extracting files"):
                zip_ref.extract(member, temp_dir)
                
        # Now find class directories and move if in expected_classes
        metadata = _load_metadata()
        expected_files = {row["video_filename"]: row["class_name"] for row in metadata}
        
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.lower().endswith(('.mov', '.mp4')) and file in expected_files:
                    class_name = expected_files[file]
                    dest_dir = RAW_VIDEO_DIR / class_name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    
                    src_file = Path(root) / file
                    dest_file = dest_dir / file
                    
                    if not dest_file.exists():
                        shutil.move(str(src_file), str(dest_file))
                        
    except Exception as e:
        logging.error(f"Error extracting {zip_path.name}: {e}")
    finally:
        # Cleanup temp dir and zip
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        zip_path.unlink()
        logging.info(f"Cleaned up {zip_path.name} and temp extraction directory.")


def verify_dataset():
    """
    Verify that all files in dataset_info.csv exist in RAW_VIDEO_DIR.
    """
    logging.info("Verifying dataset...")
    metadata = _load_metadata()
    if not metadata:
        logging.error("No metadata found. Run --metadata-only first.")
        return
        
    total_expected = len(metadata)
    found = 0
    missing = []
    class_counts = {}
    
    for row in metadata:
        class_name = row["class_name"]
        filename = row["video_filename"]
        
        expected_path = RAW_VIDEO_DIR / class_name / filename
        
        if class_name not in class_counts:
            class_counts[class_name] = {"expected": 0, "found": 0}
            
        class_counts[class_name]["expected"] += 1
        
        if expected_path.exists():
            found += 1
            class_counts[class_name]["found"] += 1
        else:
            missing.append(expected_path)
            
    logging.info(f"Verification Results: {found}/{total_expected} files found.")
    
    if missing:
        logging.warning(f"Missing {len(missing)} files.")
        if len(missing) < 20:
            for m in missing:
                logging.warning(f"  Missing: {m}")
        else:
            logging.warning("  (List truncated due to size)")
            
    print("\nPer-class summary:")
    for cls, counts in sorted(class_counts.items()):
        print(f"  {cls}: {counts['found']}/{counts['expected']}")


def main():
    parser = argparse.ArgumentParser(description="Download and organize the INCLUDE-50 dataset.")
    parser.add_argument("--metadata-only", action="store_true", help="Fetch only metadata from HuggingFace.")
    parser.add_argument("--source", choices=["zenodo", "kaggle", "local"], default="zenodo", help="Source for video downloads.")
    parser.add_argument("--local-path", type=str, help="Path to local directory containing videos (used with --source local).")
    parser.add_argument("--verify", action="store_true", help="Verify downloaded files against metadata.")
    
    args = parser.parse_args()
    
    ensure_dirs()
    
    if args.metadata_only:
        fetch_metadata()
    elif args.verify:
        verify_dataset()
    else:
        if args.source == "zenodo":
            download_from_zenodo()
        elif args.source == "kaggle":
            download_from_kaggle()
        elif args.source == "local":
            if not args.local_path:
                logging.error("--local-path is required when --source is local")
                return
            copy_from_local(args.local_path)
        else:
            logging.error("Invalid source.")


if __name__ == "__main__":
    main()
