"""
Project Configuration — Real-Time Isolated ISL Recognition
===========================================================

Central configuration shared across all modules.
Import paths, MediaPipe settings, and dataset constants from here.

Usage:
    from config import PROJECT_ROOT, DATA_DIR, MAX_HANDS, ...
"""

from pathlib import Path

# ── Project Paths ───────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.resolve()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_VIDEO_DIR = DATA_DIR / "raw_videos"
LANDMARK_DIR = DATA_DIR / "landmarks"
FEATURE_DIR = DATA_DIR / "features"
METADATA_DIR = DATA_DIR / "metadata"

# Model file (same one used by Module 1)
MODEL_PATH = PROJECT_ROOT / "hand_landmarker.task"

# Metadata files
DATASET_INFO_CSV = METADATA_DIR / "dataset_info.csv"
CLASS_MAPPING_JSON = METADATA_DIR / "class_mapping.json"

# ── MediaPipe Settings (must match Module 1 exactly) ───────────────────────

MAX_HANDS = 2
DETECTION_CONFIDENCE = 0.6
TRACKING_CONFIDENCE = 0.5

# ── Dataset Constants ──────────────────────────────────────────────────────

INCLUDE50_NUM_CLASSES = 50
INCLUDE50_NUM_SIGNERS = 7

# HuggingFace dataset identifier
HF_DATASET_ID = "ai4bharat/INCLUDE"

# ── Feature Engineering Settings ───────────────────────────────────────────

# Pad/truncate all sequences to this length (~5 seconds at 30 FPS)
MAX_SEQUENCE_LENGTH = 150

# Hand landmark indices (MediaPipe convention)
# Finger tip landmark indices
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
FINGERTIP_INDICES = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

# Finger MCP (base) landmark indices
THUMB_CMC = 1
INDEX_MCP = 5
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17
MCP_INDICES = [THUMB_CMC, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

# Wrist landmark
WRIST = 0

# Finger joint chains: each finger as a list of landmark indices from base to tip
# Used for computing joint angles and finger lengths
FINGER_CHAINS = {
    "thumb":  [0, 1, 2, 3, 4],
    "index":  [0, 5, 6, 7, 8],
    "middle": [0, 9, 10, 11, 12],
    "ring":   [0, 13, 14, 15, 16],
    "pinky":  [0, 17, 18, 19, 20],
}

# Joint triplets for angle computation: (parent, joint, child)
# The angle is measured at the middle landmark of each triplet.
JOINT_ANGLE_TRIPLETS = [
    # Thumb: 3 angles
    (0, 1, 2),    # CMC angle
    (1, 2, 3),    # MCP angle
    (2, 3, 4),    # IP angle
    # Index finger: 3 angles
    (0, 5, 6),    # MCP angle
    (5, 6, 7),    # PIP angle
    (6, 7, 8),    # DIP angle
    # Middle finger: 3 angles
    (0, 9, 10),   # MCP angle
    (9, 10, 11),  # PIP angle
    (10, 11, 12), # DIP angle
    # Ring finger: 3 angles
    (0, 13, 14),  # MCP angle
    (13, 14, 15), # PIP angle
    (14, 15, 16), # DIP angle
    # Pinky: 3 angles
    (0, 17, 18),  # MCP angle
    (17, 18, 19), # PIP angle
    (18, 19, 20), # DIP angle
]

# ── Convenience ────────────────────────────────────────────────────────────

def ensure_dirs():
    """Create all required data directories if they don't exist."""
    for d in [DATA_DIR, RAW_VIDEO_DIR, LANDMARK_DIR, FEATURE_DIR, METADATA_DIR]:
        d.mkdir(parents=True, exist_ok=True)
