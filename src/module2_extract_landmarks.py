"""
Module 2: Extract Landmarks
===========================

This script processes all downloaded INCLUDE-50 videos through MediaPipe
to extract normalized 3D hand landmarks. 

It generates:
1. .npy files containing a (num_frames, 2, 21, 3) tensor for each video.
   (Slot 0 = Right hand, Slot 1 = Left hand).
2. _meta.json files with detection statistics and video metadata.

Usage:
    python module2_extract_landmarks.py                        # Process all videos
    python module2_extract_landmarks.py --force                # Re-extract even if .npy exists
    python module2_extract_landmarks.py --class-name "Hello"   # Process one class only
    python module2_extract_landmarks.py --max-videos 5         # Process first N videos (for testing)
"""

import os
import sys
import cv2
import json
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Import shared configuration and utilities
from config import (
    DATA_DIR, 
    RAW_VIDEO_DIR, 
    LANDMARK_DIR,
    DATASET_INFO_CSV,
    ensure_dirs
)
from landmark_utils import create_hand_landmarker, extract_landmarks_from_frame

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the extraction script.
    """
    parser = argparse.ArgumentParser(description="Extract hand landmarks from videos.")
    parser.add_argument("--force", action="store_true", help="Re-extract even if .npy exists.")
    parser.add_argument("--class-name", type=str, help="Process one class only.")
    parser.add_argument("--max-videos", type=int, help="Process first N videos (for testing).")
    return parser.parse_args()

def process_video(video_path: Path, landmarker, output_npy_path: Path, output_json_path: Path, row: pd.Series) -> tuple[bool, int, int]:
    """
    Process a single video, extract landmarks for each frame, and save to disk.
    
    Parameters:
        video_path (Path): Path to the input video.
        landmarker (HandLandmarker): MediaPipe landmarker instance.
        output_npy_path (Path): Path to save the extracted landmarks.
        output_json_path (Path): Path to save the metadata JSON.
        row (pd.Series): Metadata row for this video from the CSV.
        
    Returns:
        tuple[bool, int, int]: A tuple containing:
            - success flag (True if successfully processed and saved)
            - total number of frames processed
            - number of frames where at least one hand was detected
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, 0, 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    all_frames = []
    frames_with_hands = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR (OpenCV default) to RGB (MediaPipe requirement)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract normalized landmarks using shared utility
        hands = extract_landmarks_from_frame(landmarker, rgb_frame)
        
        # Prepare frame array: shape (2, 21, 3) 
        # Slot 0: Right hand, Slot 1: Left hand
        frame_data = np.zeros((2, 21, 3), dtype=np.float32)
        
        if hands:
            frames_with_hands += 1
            for hand in hands:
                h_label = hand["handedness"]
                lm_array = hand["landmarks_array"]
                
                # Assign to correct slot based on handedness
                # If multiple hands of the same handedness are detected, the first one fills the slot.
                if h_label == "Right" and not np.any(frame_data[0]):
                    frame_data[0] = lm_array
                elif h_label == "Left" and not np.any(frame_data[1]):
                    frame_data[1] = lm_array
                
        all_frames.append(frame_data)
        
    cap.release()
    
    num_frames = len(all_frames)
    if num_frames == 0:
        return False, 0, 0
        
    hands_detected_ratio = frames_with_hands / num_frames
    
    # Save .npy file (num_frames, 2, 21, 3)
    np.save(output_npy_path, np.stack(all_frames))
    
    # Create metadata dictionary
    meta_info = {
        "num_frames": num_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "hands_detected_ratio": hands_detected_ratio,
        "signer_id": row.get("signer_id", ""),
        "split": row.get("split", ""),
        "class_name": row.get("class_name", ""),
        "class_index": int(row.get("class_index", -1))
    }
    
    # Save companion JSON metadata
    with open(output_json_path, "w") as f:
        json.dump(meta_info, f, indent=4)
        
    return True, num_frames, frames_with_hands


def main():
    """
    Main pipeline entrypoint to extract landmarks for all specified videos.
    """
    args = parse_args()
    
    # Ensure necessary directories exist
    ensure_dirs()
    
    print("Loading metadata...")
    if not DATASET_INFO_CSV.exists():
        print(f"Error: Metadata file {DATASET_INFO_CSV} not found.")
        print("Please run the dataset download module first (module2_download_dataset.py).")
        sys.exit(1)
        
    # Read metadata created by download module
    df = pd.read_csv(DATASET_INFO_CSV)
    
    # Apply CLI filters
    if args.class_name:
        df = df[df["class_name"] == args.class_name]
        print(f"Filtered to class '{args.class_name}': {len(df)} videos found.")
        
    if args.max_videos:
        df = df.head(args.max_videos)
        print(f"Limited to first {args.max_videos} videos.")
        
    print("Initializing MediaPipe HandLandmarker...")
    try:
        landmarker = create_hand_landmarker()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
        
    # Variables for summary report
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    # Track statistics per class
    class_stats = defaultdict(lambda: {"count": 0, "frames": 0, "detection_ratio_sum": 0.0})
    low_detection_videos = []
    
    start_time = time.time()
    
    try:
        # Tqdm progress bar over videos
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Landmarks", unit="video"):
            video_filename = row["video_filename"]
            class_name = row["class_name"]
            
            # Setup paths
            video_path = RAW_VIDEO_DIR / class_name / video_filename
            class_landmark_dir = LANDMARK_DIR / class_name
            class_landmark_dir.mkdir(parents=True, exist_ok=True)
            
            # Filenames without extension
            stem = Path(video_filename).stem
            output_npy_path = class_landmark_dir / f"{stem}.npy"
            output_json_path = class_landmark_dir / f"{stem}_meta.json"
            
            # Resumability check
            if output_npy_path.exists() and not args.force:
                skipped_count += 1
                
                # Attempt to read metadata for summary stats even if skipped
                if output_json_path.exists():
                    try:
                        with open(output_json_path, "r") as f:
                            meta = json.load(f)
                        c_stats = class_stats[class_name]
                        c_stats["count"] += 1
                        c_stats["frames"] += meta["num_frames"]
                        c_stats["detection_ratio_sum"] += meta["hands_detected_ratio"]
                        
                        if meta["hands_detected_ratio"] < 0.5:
                            low_detection_videos.append((video_filename, meta["hands_detected_ratio"]))
                    except Exception:
                        pass
                continue
                
            if not video_path.exists():
                tqdm.write(f"Warning: Video not found at {video_path}")
                failed_count += 1
                continue
                
            # Run the extraction on this video
            success, num_frames, frames_with_hands = process_video(
                video_path, landmarker, output_npy_path, output_json_path, row
            )
            
            if success:
                processed_count += 1
                ratio = frames_with_hands / num_frames if num_frames > 0 else 0
                
                c_stats = class_stats[class_name]
                c_stats["count"] += 1
                c_stats["frames"] += num_frames
                c_stats["detection_ratio_sum"] += ratio
                
                if ratio < 0.5:
                    low_detection_videos.append((video_filename, ratio))
            else:
                tqdm.write(f"Warning: Failed to process video {video_path} (could be corrupt or unreadable).")
                failed_count += 1
                
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Stopping gracefully...")
    finally:
        # Proper cleanup for MediaPipe instance
        landmarker.close()
        
    # Calculate elapsed time and speed
    elapsed_time = time.time() - start_time
    mins, secs = divmod(elapsed_time, 60)
    
    # ── Summary Report ──────────────────────────────────────────────────────────
    
    print("\n" + "="*55)
    print("EXTRACTION SUMMARY REPORT")
    print("="*55)
    print(f"Total time elapsed  : {int(mins)}m {int(secs)}s")
    if (processed_count) > 0:
        speed = processed_count / elapsed_time * 60
        print(f"Processing speed    : {speed:.1f} videos/min")
    print(f"Videos processed    : {processed_count}")
    print(f"Videos skipped      : {skipped_count} (already extracted)")
    print(f"Videos failed       : {failed_count}")
    
    if class_stats:
        print("\nPer-Class Statistics:")
        print(f"{'Class Name':<20} | {'Count':<6} | {'Avg Frames':<12} | {'Avg Det Ratio':<12}")
        print("-" * 55)
        for c_name, stats in sorted(class_stats.items()):
            count = stats["count"]
            if count > 0:
                avg_f = stats["frames"] / count
                avg_r = stats["detection_ratio_sum"] / count
                print(f"{c_name:<20} | {count:<6} | {avg_f:<12.1f} | {avg_r:<12.2%}")
                
    if low_detection_videos:
        print("\nWarnings: Videos with < 50% hand detection:")
        for v_name, r in low_detection_videos:
            print(f"  - {v_name} ({r:.1%})")
    
    print("="*55)

if __name__ == "__main__":
    main()
