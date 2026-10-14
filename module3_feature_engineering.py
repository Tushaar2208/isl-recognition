"""
Module 3: Feature Engineering
=============================

This script transforms raw landmark sequences into feature vectors for classification.
It computes both static features (angles, distances) and dynamic features (velocities)
for isolated sign language recognition.
"""

import argparse
import itertools
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    DATASET_INFO_CSV,
    FEATURE_DIR,
    FINGER_CHAINS,
    JOINT_ANGLE_TRIPLETS,
    LANDMARK_DIR,
    MAX_SEQUENCE_LENGTH,
)

def compute_joint_angles(landmarks_21x3: np.ndarray) -> np.ndarray:
    """
    Compute 15 joint angles based on JOINT_ANGLE_TRIPLETS.
    
    Parameters
    ----------
    landmarks_21x3 : np.ndarray
        Shape (21, 3) representing the x, y, z coordinates of hand landmarks.
        
    Returns
    -------
    np.ndarray
        Shape (15,) containing the angles in radians [0, pi].
        Returns zeros if landmarks are all zeros (no hand).
    """
    if not np.any(landmarks_21x3):
        return np.zeros(15)

    angles = []
    for a, b, c in JOINT_ANGLE_TRIPLETS:
        vec1 = landmarks_21x3[a] - landmarks_21x3[b]
        vec2 = landmarks_21x3[c] - landmarks_21x3[b]
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            angles.append(0.0)
            continue
            
        dot_prod = np.dot(vec1, vec2) / (norm1 * norm2)
        dot_prod = np.clip(dot_prod, -1.0, 1.0)
        angle = np.arccos(dot_prod)
        angles.append(angle)
        
    return np.array(angles)

def compute_fingertip_distances(landmarks_21x3: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distance between all pairs of fingertips.
    Useful for capturing fine-grained hand shapes like pinching.
    
    Parameters
    ----------
    landmarks_21x3 : np.ndarray
        Shape (21, 3)
        
    Returns
    -------
    np.ndarray
        Shape (10,) representing pairwise distances.
    """
    if not np.any(landmarks_21x3):
        return np.zeros(10)
        
    fingertips = [4, 8, 12, 16, 20]
    distances = []
    
    for idx1, idx2 in itertools.combinations(fingertips, 2):
        dist = np.linalg.norm(landmarks_21x3[idx1] - landmarks_21x3[idx2])
        distances.append(dist)
        
    return np.array(distances)

def compute_palm_distances(landmarks_21x3: np.ndarray) -> np.ndarray:
    """
    Compute distances from each fingertip to the palm center.
    Useful for detecting open vs closed hands.
    
    Parameters
    ----------
    landmarks_21x3 : np.ndarray
        Shape (21, 3)
        
    Returns
    -------
    np.ndarray
        Shape (5,)
    """
    if not np.any(landmarks_21x3):
        return np.zeros(5)
        
    # Palm center is average of wrist and 4 MCP joints
    palm_center = np.mean(landmarks_21x3[[0, 5, 9, 13, 17]], axis=0)
    fingertips = [4, 8, 12, 16, 20]
    
    distances = [np.linalg.norm(landmarks_21x3[i] - palm_center) for i in fingertips]
    return np.array(distances)

def compute_extension_ratios(landmarks_21x3: np.ndarray) -> np.ndarray:
    """
    Compute extension ratio for each finger.
    Ratio = dist(fingertip, finger_MCP) / total_bone_chain_length.
    Captures how curled a finger is (1.0 = extended, 0.0 = curled).
    
    Parameters
    ----------
    landmarks_21x3 : np.ndarray
        Shape (21, 3)
        
    Returns
    -------
    np.ndarray
        Shape (5,)
    """
    if not np.any(landmarks_21x3):
        return np.zeros(5)
        
    ratios = []
    # Process in standard order: thumb, index, middle, ring, pinky
    for finger in ["thumb", "index", "middle", "ring", "pinky"]:
        chain = FINGER_CHAINS[finger]
        tip_idx = chain[-1]
        mcp_idx = chain[1]
        
        straight_dist = np.linalg.norm(landmarks_21x3[tip_idx] - landmarks_21x3[mcp_idx])
        
        # Sum consecutive bone lengths
        chain_len = 0.0
        for i in range(len(chain) - 1):
            chain_len += np.linalg.norm(landmarks_21x3[chain[i+1]] - landmarks_21x3[chain[i]])
            
        if chain_len < 1e-6:
            ratios.append(0.0)
        else:
            ratios.append(straight_dist / chain_len)
            
    return np.array(ratios)

def compute_thumb_distances(landmarks_21x3: np.ndarray) -> np.ndarray:
    """
    Compute distance from thumb tip to other fingertips.
    Crucial for detecting numerical signs or specific pinch configurations.
    
    Parameters
    ----------
    landmarks_21x3 : np.ndarray
        Shape (21, 3)
        
    Returns
    -------
    np.ndarray
        Shape (4,)
    """
    if not np.any(landmarks_21x3):
        return np.zeros(4)
        
    thumb_tip = landmarks_21x3[4]
    other_tips = [8, 12, 16, 20]
    
    distances = [np.linalg.norm(thumb_tip - landmarks_21x3[i]) for i in other_tips]
    return np.array(distances)

def compute_hand_orientation(landmarks_21x3: np.ndarray) -> np.ndarray:
    """
    Compute hand orientation (pitch and yaw) using palm normal vector.
    Helps distinguish signs that differ only in hand rotation.
    
    Parameters
    ----------
    landmarks_21x3 : np.ndarray
        Shape (21, 3)
        
    Returns
    -------
    np.ndarray
        Shape (2,) - [pitch, yaw]
    """
    if not np.any(landmarks_21x3):
        return np.zeros(2)
        
    vec1 = landmarks_21x3[5] - landmarks_21x3[0]  # Wrist to Index MCP
    vec2 = landmarks_21x3[17] - landmarks_21x3[0] # Wrist to Pinky MCP
    
    normal = np.cross(vec1, vec2)
    norm_len = np.linalg.norm(normal)
    
    if norm_len < 1e-6:
        return np.zeros(2)
        
    normal = normal / norm_len
    nx, ny, nz = normal
    
    pitch = np.arcsin(np.clip(ny, -1.0, 1.0))
    yaw = np.arctan2(nx, nz)
    
    return np.array([pitch, yaw])

def compute_static_features(landmarks_21x3: np.ndarray) -> np.ndarray:
    """
    Compute all static features for a single hand in a single frame.
    
    Parameters
    ----------
    landmarks_21x3 : np.ndarray
        Shape (21, 3)
        
    Returns
    -------
    np.ndarray
        Shape (41,) - concatenated static features
    """
    if not np.any(landmarks_21x3):
        return np.zeros(41)
        
    f1 = compute_joint_angles(landmarks_21x3)         # 15
    f2 = compute_fingertip_distances(landmarks_21x3)  # 10
    f3 = compute_palm_distances(landmarks_21x3)       # 5
    f4 = compute_extension_ratios(landmarks_21x3)     # 5
    f5 = compute_thumb_distances(landmarks_21x3)      # 4
    f6 = compute_hand_orientation(landmarks_21x3)     # 2
    
    features = np.concatenate([f1, f2, f3, f4, f5, f6])
    
    # Replace NaNs or Infs from calculation edge cases
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features

def compute_velocities(sequence_Tx21x3: np.ndarray, mode: str = "compact") -> np.ndarray:
    """
    Compute velocity features across frames for a sequence.
    
    Parameters
    ----------
    sequence_Tx21x3 : np.ndarray
        Shape (T, 21, 3) - Sequence of landmarks for one hand.
    mode : str
        "compact" for 6 key points (wrist + 5 tips), "full" for all 21 points.
        
    Returns
    -------
    np.ndarray
        Shape (T, 19) or (T, 64) depending on mode.
        Includes motion energy as the last feature.
    """
    T = sequence_Tx21x3.shape[0]
    
    if mode == "compact":
        indices = [0, 4, 8, 12, 16, 20]
    else:
        indices = list(range(21))
        
    seq = sequence_Tx21x3[:, indices, :] # (T, K, 3)
    vel = np.zeros_like(seq)
    
    if T > 1:
        vel[1:] = seq[1:] - seq[:-1]
        
    vel_flat = vel.reshape(T, -1) # (T, K*3)
    energy = np.sum(vel_flat**2, axis=1, keepdims=True) # (T, 1)
    
    return np.concatenate([vel_flat, energy], axis=1)

def compute_frame_features(landmarks_2x21x3: np.ndarray, prev_landmarks: np.ndarray = None, mode: str = "compact") -> np.ndarray:
    """
    Compute feature vector for a single frame, combining static and velocity features
    for both hands.
    
    Parameters
    ----------
    landmarks_2x21x3 : np.ndarray
        Shape (2, 21, 3) - Current frame landmarks for right and left hands.
    prev_landmarks : np.ndarray, optional
        Shape (2, 21, 3) - Previous frame landmarks.
    mode : str
        "compact" or "full".
        
    Returns
    -------
    np.ndarray
        Shape (120,) for compact or (210,) for full.
    """
    feats = []
    
    for hand_idx in range(2):
        hand = landmarks_2x21x3[hand_idx]
        static = compute_static_features(hand)
        
        if mode == "compact":
            indices = [0, 4, 8, 12, 16, 20]
        else:
            indices = list(range(21))
            
        if prev_landmarks is None:
            vel_flat = np.zeros(len(indices) * 3)
            energy = np.array([0.0])
        else:
            prev_hand = prev_landmarks[hand_idx]
            vel = hand[indices] - prev_hand[indices]
            vel_flat = vel.flatten()
            energy = np.array([np.sum(vel_flat**2)])
            
        vel_feats = np.concatenate([vel_flat, energy])
        feats.append(np.concatenate([static, vel_feats]))
        
    return np.concatenate(feats)

def process_video_features(landmark_path: Path, mode: str = "compact") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process an entire video's landmark sequence into feature representations.
    Extracts frame features, pads/truncates to MAX_SEQUENCE_LENGTH, and computes summary stats.
    
    Parameters
    ----------
    landmark_path : Path
        Path to .npy file of shape (T, 2, 21, 3)
    mode : str
        "compact" or "full"
        
    Returns
    -------
    tuple
        features: (MAX_SEQUENCE_LENGTH, D) - padded feature sequence
        mask: (MAX_SEQUENCE_LENGTH,) - binary mask of valid frames
        stats: (D * 4,) - mean, std, min, max across time
    """
    video_data = np.load(landmark_path) # (T, 2, 21, 3)
    T = video_data.shape[0]
    
    # Compute features for all frames
    frame_feats = []
    prev = None
    for t in range(T):
        feat = compute_frame_features(video_data[t], prev, mode=mode)
        frame_feats.append(feat)
        prev = video_data[t]
        
    if len(frame_feats) == 0:
        # Edge case: empty sequence
        feat_dim = 120 if mode == "compact" else 210
        return np.zeros((MAX_SEQUENCE_LENGTH, feat_dim)), np.zeros(MAX_SEQUENCE_LENGTH, dtype=int), np.zeros(feat_dim * 4)
        
    frame_feats = np.array(frame_feats) # (T, D)
    feat_dim = frame_feats.shape[1]
    
    # Pad or truncate
    if T >= MAX_SEQUENCE_LENGTH:
        padded_feats = frame_feats[:MAX_SEQUENCE_LENGTH]
        mask = np.ones(MAX_SEQUENCE_LENGTH, dtype=int)
        valid_feats = padded_feats
    else:
        padded_feats = np.zeros((MAX_SEQUENCE_LENGTH, feat_dim))
        padded_feats[:T] = frame_feats
        mask = np.zeros(MAX_SEQUENCE_LENGTH, dtype=int)
        mask[:T] = 1
        valid_feats = frame_feats
        
    # Aggregate statistics (mean, std, min, max per feature)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_f = np.mean(valid_feats, axis=0)
        std_f = np.std(valid_feats, axis=0)
        min_f = np.min(valid_feats, axis=0)
        max_f = np.max(valid_feats, axis=0)
        
    stats = np.concatenate([mean_f, std_f, min_f, max_f])
    stats = np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)
    
    return padded_feats, mask, stats

def main():
    parser = argparse.ArgumentParser(description="Feature Engineering for ISL")
    parser.add_argument("--mode", type=str, choices=["compact", "full"], default="compact", help="Feature extraction mode")
    parser.add_argument("--class-name", type=str, default=None, help="Process single class")
    parser.add_argument("--report", action="store_true", help="Generate summary report")
    args = parser.parse_args()
    
    if not DATASET_INFO_CSV.exists():
        print(f"Dataset info not found at {DATASET_INFO_CSV}. Please run Module 2 first.")
        return
        
    df = pd.read_csv(DATASET_INFO_CSV)
    
    if args.class_name:
        df = df[df['class_name'] == args.class_name]
        if df.empty:
            print(f"Class {args.class_name} not found in metadata.")
            return
            
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    
    report_data = []
    
    for class_name, group in df.groupby("class_name"):
        print(f"Processing class: {class_name}")
        class_feature_dir = FEATURE_DIR / class_name
        class_feature_dir.mkdir(exist_ok=True)
        
        valid_frames_list = []
        
        for _, row in tqdm(group.iterrows(), total=len(group)):
            # video_filename in CSV includes extension (e.g., "MVI_1234.MOV")
            # The landmark .npy files use the stem (e.g., "MVI_1234.npy")
            video_stem = Path(row['video_filename']).stem
            npy_path = LANDMARK_DIR / class_name / f"{video_stem}.npy"
            
            if not npy_path.exists():
                continue
                
            features, mask, stats = process_video_features(npy_path, mode=args.mode)
            
            # Save features
            np.save(class_feature_dir / f"{video_stem}_features.npy", features)
            np.save(class_feature_dir / f"{video_stem}_mask.npy", mask)
            np.save(class_feature_dir / f"{video_stem}_stats.npy", stats)
            
            valid_frames = np.sum(mask)
            valid_frames_list.append(valid_frames)
            
        if valid_frames_list:
            avg_seq_len = np.mean(valid_frames_list)
            avg_pad_ratio = 1.0 - (avg_seq_len / MAX_SEQUENCE_LENGTH)
            report_data.append({
                "Class": class_name,
                "Count": len(valid_frames_list),
                "Avg_Seq_Len": avg_seq_len,
                "Avg_Pad_Ratio": avg_pad_ratio
            })
            
    if args.report or report_data:
        feat_dim = 120 if args.mode == "compact" else 210
        report_str = f"Feature Engineering Report\n"
        report_str += f"==========================\n"
        report_str += f"Mode: {args.mode}\n"
        report_str += f"Feature Dimension (Per Frame): {feat_dim}\n"
        report_str += f"Stats Dimension (Per Video): {feat_dim * 4}\n\n"
        
        report_str += f"{'Class':<20} {'Count':<10} {'Avg Seq Len':<15} {'Avg Pad Ratio':<15}\n"
        report_str += "-" * 65 + "\n"
        for r in report_data:
            report_str += f"{r['Class']:<20} {r['Count']:<10} {r['Avg_Seq_Len']:<15.2f} {r['Avg_Pad_Ratio']:<15.2f}\n"
            
        report_path = FEATURE_DIR / "feature_report.txt"
        with open(report_path, "w") as f:
            f.write(report_str)
            
        print(f"\nReport saved to {report_path}")
        print(report_str)

if __name__ == "__main__":
    main()
