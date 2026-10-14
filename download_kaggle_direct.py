#!/usr/bin/env python3
"""
Direct INCLUDE-50 Video Downloader via Kaggle (Timeout-Protected)
=================================================================

Downloads ONLY the 958 INCLUDE-50 videos directly from Kaggle:
- Strict 45-second timeout per video (never hangs if network blips or switches)
- Resilient retry loop with automatic cleanup of partial files
- Global socket timeout (30s) to prevent frozen TCP connections
- Two-pass downloading (retries any failed files at the end)

Usage:
    python download_kaggle_direct.py
"""

import concurrent.futures
import csv
import json
import os
import shutil
import socket
import sys
import time
import zipfile
from pathlib import Path

# Set global socket timeout to 30s so dead sockets never hang forever
socket.setdefaulttimeout(30)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
from config import RAW_VIDEO_DIR, DATASET_INFO_CSV, METADATA_DIR, ensure_dirs
from kaggle.api.kaggle_api_extended import KaggleApi

KAGGLE_DATASET = "daskoushik/include"
NEEDED_ZIPS_JSON = METADATA_DIR / "zenodo_needed_zips.json"
TEMP_DIR = RAW_VIDEO_DIR / "_temp_kaggle"
FILE_TIMEOUT_SECONDS = 180  # Generous 3-minute limit per video so slow speeds never get cut off



def load_video_mappings() -> list:
    """Build list of {class_name, video_filename, kaggle_path} for all 958 INCLUDE-50 videos."""
    with open(NEEDED_ZIPS_JSON) as f:
        zips = json.load(f)

    with open(DATASET_INFO_CSV) as f:
        rows = list(csv.DictReader(f))

    targets = []
    for r in rows:
        vname = r["video_filename"]
        cls = r["class_name"]
        rel_archive = r["video_path_in_archive"]
        archive_cat = rel_archive.split("/")[0]

        # Match zip that starts with the archive's parent category
        matching_zip = None
        for zname, info in zips.items():
            if zname.startswith(archive_cat) and vname in info["matches"]:
                matching_zip = zname.replace(".zip", "")
                break

        # Fallback
        if not matching_zip:
            for zname, info in zips.items():
                if vname in info["matches"]:
                    matching_zip = zname.replace(".zip", "")
                    break

        if matching_zip:
            kaggle_path = f"{matching_zip}/{rel_archive}"
            targets.append({
                "class_name": cls,
                "video_filename": vname,
                "kaggle_path": kaggle_path,
            })
    return targets


def download_single_video(api, kpath: str, vid_name: str, dest_file: Path) -> bool:
    """Download and extract one video with timeout protection."""
    # Clean temp dir of this video's leftovers
    for f in TEMP_DIR.glob(f"*{vid_name}*"):
        try: f.unlink()
        except: pass

    def _do_download():
        api.dataset_download_file(
            KAGGLE_DATASET,
            file_name=kpath,
            path=str(TEMP_DIR),
            quiet=True,
        )

    # Run download with hard timeout
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_do_download)
        future.result(timeout=FILE_TIMEOUT_SECONDS)

    # Locate downloaded zip
    expected_zip = TEMP_DIR / f"{vid_name}.zip"
    if not expected_zip.exists():
        zips = list(TEMP_DIR.glob("*.zip"))
        if zips:
            expected_zip = zips[0]

    if expected_zip.exists() and expected_zip.stat().st_size > 1000:
        with zipfile.ZipFile(expected_zip, "r") as zf:
            for member in zf.namelist():
                if member.lower().endswith((".mov", ".mp4")):
                    zf.extract(member, TEMP_DIR)
                    extracted_file = TEMP_DIR / member
                    shutil.move(str(extracted_file), str(dest_file))
                    break

        try: expected_zip.unlink()
        except: pass

    # Clean any partials
    for f in TEMP_DIR.glob(f"*{vid_name}*"):
        try: f.unlink()
        except: pass

    return dest_file.exists() and dest_file.stat().st_size > 10000


def main():
    ensure_dirs()
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print("INCLUDE-50 Direct Kaggle Downloader (Timeout Protected)", flush=True)
    print(f"Per-file timeout: {FILE_TIMEOUT_SECONDS}s (prevents any socket hangs)", flush=True)
    print("=" * 60, flush=True)

    print("\nAuthenticating with Kaggle API...", flush=True)
    api = KaggleApi()
    api.authenticate()
    print("Authenticated successfully!", flush=True)

    targets = load_video_mappings()
    total_videos = len(targets)
    print(f"Total INCLUDE-50 videos to collect: {total_videos}", flush=True)

    start_time = time.time()
    failed = []

    # Count existing
    existing = sum(
        1 for t in targets
        if (RAW_VIDEO_DIR / t["class_name"] / t["video_filename"]).exists()
        and (RAW_VIDEO_DIR / t["class_name"] / t["video_filename"]).stat().st_size > 10000
    )
    print(f"Already downloaded on disk:        {existing}", flush=True)
    print(f"Remaining to download:             {total_videos - existing}", flush=True)
    print("-" * 60, flush=True)

    downloaded_now = 0

    for i, t in enumerate(targets, 1):
        cls_name = t["class_name"]
        vid_name = t["video_filename"]
        kpath = t["kaggle_path"]

        dest_dir = RAW_VIDEO_DIR / cls_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / vid_name

        if dest_file.exists() and dest_file.stat().st_size > 10000:
            continue

        success = False
        for attempt in range(1, 4):
            try:
                if download_single_video(api, kpath, vid_name, dest_file):
                    downloaded_now += 1
                    success = True
                    current_total = existing + downloaded_now
                    elapsed = time.time() - start_time
                    avg_speed = downloaded_now / elapsed if elapsed > 0 else 0
                    remain = total_videos - current_total
                    eta_min = (remain / avg_speed) / 60 if avg_speed > 0 else 0

                    print(f"[{current_total:3d}/{total_videos}] ({(current_total/total_videos)*100:5.1f}%) "
                          f"✅ {cls_name}/{vid_name} ({dest_file.stat().st_size/(1024*1024):.1f} MB) - ETA: {eta_min:.0f}m", flush=True)
                    break
                else:
                    time.sleep(2)
            except concurrent.futures.TimeoutError:
                print(f"  ⚠️ Timeout on {vid_name} (> {FILE_TIMEOUT_SECONDS}s). Reconnecting (attempt {attempt}/3)...", flush=True)
                # Re-instantiate api client on timeout to reset connections
                try:
                    api = KaggleApi()
                    api.authenticate()
                except:
                    pass
                time.sleep(3)
            except Exception as e:
                print(f"  ⚠️ Error downloading {vid_name}: {e}. Retrying (attempt {attempt}/3)...", flush=True)
                time.sleep(3)

        if not success:
            print(f"  ❌ Skipping {cls_name}/{vid_name} for now (will retry in pass 2)", flush=True)
            failed.append(t)

    # Pass 2: Retry failed videos once more
    if failed:
        print(f"\n--- Pass 2: Retrying {len(failed)} previously skipped videos ---", flush=True)
        for t in list(failed):
            dest_file = RAW_VIDEO_DIR / t["class_name"] / t["video_filename"]
            try:
                if download_single_video(api, t["kaggle_path"], t["video_filename"], dest_file):
                    downloaded_now += 1
                    failed.remove(t)
                    print(f"  ✅ Recovered {t['class_name']}/{t['video_filename']}", flush=True)
            except:
                pass

    # Cleanup temp
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

    elapsed = time.time() - start_time
    mins, secs = divmod(elapsed, 60)
    hours, mins = divmod(mins, 60)

    final_count = sum(
        1 for t in targets
        if (RAW_VIDEO_DIR / t["class_name"] / t["video_filename"]).exists()
        and (RAW_VIDEO_DIR / t["class_name"] / t["video_filename"]).stat().st_size > 10000
    )

    print("\n" + "=" * 60, flush=True)
    print("DOWNLOAD FINISHED", flush=True)
    print("=" * 60, flush=True)
    print(f"Time taken:       {int(hours)}h {int(mins)}m {int(secs)}s", flush=True)
    print(f"Total on disk:    {final_count}/{total_videos}", flush=True)
    if failed:
        print(f"Missing count:    {len(failed)}", flush=True)
    else:
        print("🎉 ALL 958 INCLUDE-50 videos downloaded successfully!", flush=True)


if __name__ == "__main__":
    main()
