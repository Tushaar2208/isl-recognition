#!/usr/bin/env python3
"""
Real-Time Isolated Indian Sign Language (ISL) Recognition
==========================================================
Main Project Entrypoint (B.Tech CSE Final Year Project)

Usage:
    python main.py --module 1      # Live webcam sign capture & landmark overlay
    python main.py --module 2      # Batch video landmark extraction (MediaPipe)
    python main.py --module 3      # Geometric feature engineering
    python main.py --module 4      # Train & evaluate classifiers (RF & SVM)
    python main.py --demo          # Visualize extracted landmarks on dataset video
    python main.py --status        # Show pipeline status and data statistics
"""

import argparse
import sys
from pathlib import Path

# Add project root and src/ to search path
ROOT_DIR = Path(__file__).parent.resolve()
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    DATA_DIR, RAW_VIDEO_DIR, LANDMARK_DIR,
    FEATURE_DIR, METADATA_DIR, MODEL_PATH
)


def print_banner():
    print("=" * 65)
    print("  Real-Time Isolated Indian Sign Language (ISL) Recognition")
    print("  B.Tech CSE Final Year Project — AKTU")
    print("=" * 65)


def show_status():
    print_banner()
    print("\n📊 Pipeline Status & Dataset Metrics:")
    print("-" * 65)

    # Videos
    raw_vids = list(RAW_VIDEO_DIR.rglob("*.MOV")) + list(RAW_VIDEO_DIR.rglob("*.MP4")) if RAW_VIDEO_DIR.exists() else []
    print(f"  [Module 2] Raw Videos Downloaded  : {len(raw_vids)} / 958 ({sum(f.stat().st_size for f in raw_vids)/(1024**3):.2f} GB)" if raw_vids else "  [Module 2] Raw Videos Downloaded  : Not found")

    # Landmarks
    npys = list(LANDMARK_DIR.rglob("*.npy")) if LANDMARK_DIR.exists() else []
    print(f"  [Module 2] Extracted Landmarks    : {len(npys)} / 958 files" if npys else "  [Module 2] Extracted Landmarks    : None")

    # Features
    feats = list(FEATURE_DIR.glob("*/*_stats.npy")) if FEATURE_DIR.exists() else []
    print(f"  [Module 3] Engineered Features    : {len(feats)} / 958 videos" if feats else "  [Module 3] Engineered Features    : None")


    # Models
    rf_model = DATA_DIR / "models" / "random_forest.pkl"
    eval_rep = DATA_DIR / "models" / "evaluation_report.txt"
    if eval_rep.exists():
        with open(eval_rep) as f:
            lines = [l.strip() for l in f.readlines() if ":" in l and "%" in l]
        results_str = ", ".join(lines[:2])
        print(f"  [Module 4] Trained Classifiers   : Ready ({results_str})")
    elif rf_model.exists():
        print(f"  [Module 4] Trained Classifiers   : Ready")
    else:
        print(f"  [Module 4] Trained Classifiers   : Not yet trained")

    print("-" * 65)
    print("\n🚀 Available Commands:")
    print("  python main.py --module 1   # Live webcam capture & landmark detection")
    print("  python main.py --module 2   # Batch landmark extraction on dataset")
    print("  python main.py --module 3   # Feature engineering from landmarks")
    print("  python main.py --module 4   # Train classifiers & plot confusion matrix")
    print("  python main.py --demo       # Video landmark overlay visualizer")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="Indian Sign Language Recognition System CLI"
    )
    parser.add_argument(
        "--module", type=int, choices=[1, 2, 3, 4],
        help="Run specific module (1: Capture, 2: Extraction, 3: Features, 4: Classifiers)"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run visual landmark demonstration on dataset video"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Display pipeline status and summary statistics"
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to specific video file to visualize with --demo"
    )

    args = parser.parse_args()

    if args.module == 1:
        print_banner()
        print("\nStarting Module 1: Live Sign Capture (Press 'q' to quit)...")
        from src.module1_sign_capture import main as m1_main
        m1_main()

    elif args.module == 2:
        print_banner()
        print("\nStarting Module 2: Dataset Landmark Extraction...")
        from src.module2_extract_landmarks import main as m2_main
        m2_main()

    elif args.module == 3:
        print_banner()
        print("\nStarting Module 3: Geometric Feature Engineering...")
        from src.module3_feature_engineering import main as m3_main
        m3_main()

    elif args.module == 4:
        print_banner()
        print("\nStarting Module 4: Classifier Training & Evaluation...")
        from src.module4_train_classifiers import main as m4_main
        m4_main()

    elif args.demo:
        print_banner()
        print("\nStarting Landmark Visualizer (Press 'q' to quit, SPACE to pause)...")
        from src.visualize_video_landmarks import main as demo_main
        demo_main(video=args.video)

    else:
        show_status()



if __name__ == "__main__":
    main()
