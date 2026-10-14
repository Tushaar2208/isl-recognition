#!/usr/bin/env python3
"""
Visualize Extracted Landmarks on Video
======================================
Plays a video side-by-side with its extracted landmarks overlaid.
Perfect for viva / presentation demonstrations!

Usage:
    python visualize_video_landmarks.py
    python visualize_video_landmarks.py --video "data/raw_videos/1. loud/MVI_9448.MOV"
"""

import argparse
import cv2
import numpy as np
from pathlib import Path

# MediaPipe hand connections (pairs of landmark indices)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index
    (5, 9), (9, 10), (10, 11), (11, 12),      # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # Pinky
    (0, 17)                                   # Palm base
]


def draw_skeleton(frame, landmarks, color=(0, 255, 0)):
    """Draw 21 landmarks and connecting bones on frame (landmarks in pixel coords)."""
    h, w = frame.shape[:2]
    # Check if hand is detected (not all zeros)
    if not np.any(landmarks):
        return

    # Draw connections
    for start_idx, end_idx in HAND_CONNECTIONS:
        pt1 = (int(landmarks[start_idx, 0] * w), int(landmarks[start_idx, 1] * h))
        pt2 = (int(landmarks[end_idx, 0] * w), int(landmarks[end_idx, 1] * h))
        cv2.line(frame, pt1, pt2, color, 2)

    # Draw landmark points
    for idx in range(21):
        cx = int(landmarks[idx, 0] * w)
        cy = int(landmarks[idx, 1] * h)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    args = parser.parse_args()

    # Default to first video found if none specified
    if args.video:
        vid_path = Path(args.video)
    else:
        vids = list(Path("data/raw_videos").rglob("*.MOV")) + list(Path("data/raw_videos").rglob("*.MP4"))
        if not vids:
            print("No videos found in data/raw_videos/")
            return
        vid_path = vids[0]

    # Find companion .npy landmark file
    cls_name = vid_path.parent.name
    npy_path = Path("data/landmarks") / cls_name / f"{vid_path.stem}.npy"

    if not npy_path.exists():
        print(f"Landmark file not found at {npy_path}")
        return

    print("=" * 60)
    print("  Visualizing Landmarks for ISL Video")
    print(f"  Video    : {vid_path}")
    print(f"  Landmarks: {npy_path}")
    print("  Press 'q' or 'ESC' to exit, 'SPACE' to pause/resume")
    print("=" * 60)

    cap = cv2.VideoCapture(str(vid_path))
    landmarks_seq = np.load(npy_path)  # (T, 2, 21, 3)

    frame_idx = 0
    paused = False

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret or frame_idx >= len(landmarks_seq):
                # Loop back to beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                continue

            display_frame = frame.copy()
            h, w = frame.shape[:2]

            # In the npy file, landmarks are normalized.
            # To draw them back in image space for visualization, we un-center them relative to wrist or use MediaPipe raw
            # For overlay on the original video:
            right_hand = landmarks_seq[frame_idx, 0]  # (21, 3)
            left_hand = landmarks_seq[frame_idx, 1]   # (21, 3)

            # Draw status text
            cv2.putText(display_frame, f"Class: {cls_name}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(display_frame, f"Frame: {frame_idx + 1}/{len(landmarks_seq)}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            r_det = "Right: Yes" if np.any(right_hand) else "Right: No"
            l_det = "Left: Yes" if np.any(left_hand) else "Left: No"
            cv2.putText(display_frame, f"{r_det} | {l_det}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Resize for display if video is 1080p
            scaled_frame = cv2.resize(display_frame, (960, 540))
            cv2.imshow("Module 2 - Video Landmark Visualizer", scaled_frame)
            frame_idx += 1

        key = cv2.waitKey(30) & 0xFF
        if key in [ord("q"), 27]:
            break
        elif key == ord(" "):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
