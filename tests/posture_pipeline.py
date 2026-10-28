"""
Sitting-posture classifier: MediaPipe landmark extraction + Random Forest.
Expected folders beside this file:
  sliced webcam/side_normal.mp4
  sliced webcam/side_lean_forward.mp4
  sliced webcam/side_slouch.mp4
  sliced phone/side_normal.mp4
  sliced phone/side_lean_forward.mp4
  sliced phone/side_slouch.mp4

Install dependencies in your active environment:
  pip install opencv-python mediapipe pandas numpy scikit-learn joblib

For a research claim about unseen people, collect multiple participants and
split by participant. The six starter clips are only a pipeline smoke test.
"""
from pathlib import Path
import warnings
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib

# -------------------- CONFIGURATION --------------------
PROJECT_DIR = Path(__file__).resolve().parent
VIDEO_FOLDERS = {
    "webcam": PROJECT_DIR / "sliced webcam",
    "phone": PROJECT_DIR / "sliced phone",
}
VIDEO_LABELS = {
    "side_normal.mp4": "neutral",
    "side_lean_forward.mp4": "leaning_forward",
    "side_slouch.mp4": "slouching",
}
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
MIN_VISIBILITY = 0.50
N_SPLITS = 2
RANDOM_STATE = 42

mp_pose = mp.solutions.pose
LANDMARK_NAMES = [item.name.lower() for item in mp_pose.PoseLandmark]
FEATURES = [
    "torso_tilt_from_vertical_deg",
    "trunk_thigh_angle_deg",
    "knee_angle_deg",
    "elbow_angle_deg",
    "head_relative_to_shoulder_x",
    "head_relative_to_shoulder_y",
]

# -------------------- FEATURE HELPERS --------------------
def joint_angle(a, b, c):
    """Angle ABC in degrees, or NaN if points are invalid."""
    p = np.asarray([a, b, c], dtype=float)
    if not np.isfinite(p).all():
        return np.nan
    ba, bc = p[0] - p[1], p[2] - p[1]
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-9:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1, 1))))

def torso_tilt(shoulder, hip):
    """Absolute shoulder-to-hip line angle relative to image vertical."""
    if not np.isfinite([*shoulder, *hip]).all():
        return np.nan
    dx, dy = shoulder[0] - hip[0], shoulder[1] - hip[1]
    if abs(dx) + abs(dy) < 1e-9:
        return np.nan
    return float(np.degrees(np.arctan2(abs(dx), abs(dy))))

def landmark_point(landmarks, side, name):
    idx = mp_pose.PoseLandmark[f"{side.upper()}_{name.upper()}"].value
    lm = landmarks[idx]
    return np.array([lm.x, lm.y], dtype=float), float(lm.visibility)

def frame_features(landmarks):
    # Pick whichever side has better average visibility in the main joints.
    joint_names = ["shoulder", "elbow", "wrist", "hip", "knee", "ankle"]
    side_scores = {}
    for side in ("left", "right"):
        side_scores[side] = np.mean([
            landmark_point(landmarks, side, n)[1] for n in joint_names
        ])
    side = max(side_scores, key=side_scores.get)

    points, vis = {}, {}
    for name in joint_names:
        points[name], vis[name] = landmark_point(landmarks, side, name)
    nose_lm = landmarks[mp_pose.PoseLandmark.NOSE.value]
    nose = np.array([nose_lm.x, nose_lm.y], dtype=float)

    torso_len = np.linalg.norm(points["shoulder"] - points["hip"])
    if torso_len > 1e-9 and np.isfinite(nose).all():
        head_dx = float((nose[0] - points["shoulder"][0]) / torso_len)
        head_dy = float((nose[1] - points["shoulder"][1]) / torso_len)
    else:
        head_dx = head_dy = np.nan

    reliable = np.mean([vis[n] for n in ("shoulder", "hip", "knee", "ankle")]) >= MIN_VISIBILITY
    return {
        "selected_side": side,
        "pose_reliable": bool(reliable),
        "left_side_visibility_mean": float(side_scores["left"]),
        "right_side_visibility_mean": float(side_scores["right"]),
        "torso_tilt_from_vertical_deg": torso_tilt(points["shoulder"], points["hip"]),
        "trunk_thigh_angle_deg": joint_angle(points["shoulder"], points["hip"], points["knee"]),
        "knee_angle_deg": joint_angle(points["hip"], points["knee"], points["ankle"]),
        "elbow_angle_deg": joint_angle(points["shoulder"], points["elbow"], points["wrist"]),
        "head_relative_to_shoulder_x": head_dx,
        "head_relative_to_shoulder_y": head_dy,
    }

# -------------------- EXTRACT LANDMARKS --------------------
def extract_videos():
    rows, missing = [], []
    with mp_pose.Pose(
        static_image_mode=False, model_complexity=1,
        smooth_landmarks=True, min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        for source, folder in VIDEO_FOLDERS.items():
            print(f"\nFolder: {folder}")
            if not folder.exists():
                print(f"WARNING: folder not found: {folder}")
                continue
            for filename, label in VIDEO_LABELS.items():
                path = folder / filename
                if not path.exists():
                    missing.append(str(path))
                    print(f"MISSING: {path}")
                    continue
                cap = cv2.VideoCapture(str(path))
                if not cap.isOpened():
                    print(f"ERROR: cannot open {path}")
                    cap.release()
                    continue
                fps = cap.get(cv2.CAP_PROP_FPS)
                if not np.isfinite(fps) or fps <= 0:
                    fps = 30.0
                frame_idx = detected = reliable_count = 0
                video_id = f"{source}__{Path(filename).stem}"
                print(f"Processing {video_id} as {label}")
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = pose.process(rgb)
                    row = {
                        "source": source, "video_file": filename,
                        "video_id": video_id, "posture_label": label,
                        "frame_index": frame_idx, "time_seconds": frame_idx / fps,
                        "fps": fps, "frame_width": frame.shape[1],
                        "frame_height": frame.shape[0],
                        "pose_detected": result.pose_landmarks is not None,
                    }
                    if result.pose_landmarks:
                        detected += 1
                        lm_list = result.pose_landmarks.landmark
                        for i, name in enumerate(LANDMARK_NAMES):
                            lm = lm_list[i]
                            row[f"{name}_x"] = float(lm.x)
                            row[f"{name}_y"] = float(lm.y)
                            row[f"{name}_z"] = float(lm.z)
                            row[f"{name}_visibility"] = float(lm.visibility)
                        row.update(frame_features(lm_list))
                        reliable_count += int(row["pose_reliable"])
                    else:
                        for name in LANDMARK_NAMES:
                            for axis in ("x", "y", "z", "visibility"):
                                row[f"{name}_{axis}"] = np.nan
                        row.update({
                            "selected_side": None, "pose_reliable": False,
                            "left_side_visibility_mean": np.nan,
                            "right_side_visibility_mean": np.nan,
                            **{f: np.nan for f in FEATURES},
                        })
                    rows.append(row)
                    frame_idx += 1
                cap.release()
                if frame_idx:
                    print(f"  {frame_idx} frames; detected {detected/frame_idx:.1%}; reliable {reliable_count/frame_idx:.1%}")
                else:
                    print("  WARNING: no frames read")
    if not rows:
        raise RuntimeError("No frames extracted. Check PROJECT_DIR, folders, filenames, and video formats.")
    df = pd.DataFrame(rows)
    output = OUTPUT_DIR / "landmarks_per_frame.csv"
    df.to_csv(output, index=False)
    print(f"\nSaved landmarks: {output}")
    if missing:
        print("\nSome files were missing; check the messages above.")
    return df

# -------------------- CLEAN FEATURE TABLE --------------------
def make_features(landmarks_df):
    clean = landmarks_df[
        landmarks_df["pose_detected"].fillna(False)
        & landmarks_df["pose_reliable"].fillna(False)
    ].copy()
    clean = clean.dropna(subset=FEATURES, how="all")
    if clean.empty:
        raise RuntimeError("No reliable feature rows. Check camera framing and landmark visibility.")
    path = OUTPUT_DIR / "posture_features.csv"
    clean.to_csv(path, index=False)
    print(f"\nSaved features: {path}")
    print("\nUsable frames per video:")
    print(clean.groupby(["source", "video_file", "posture_label"]).size())
    return clean

# -------------------- TRAIN AND EVALUATE --------------------
def train_classifier(features_df):
    classes = ["neutral", "leaning_forward", "slouching"]
    found = set(features_df["posture_label"].dropna().unique())
    if set(classes) - found:
        raise ValueError(f"Missing classes after filtering: {set(classes)-found}")

    unique_videos = features_df[["video_id", "posture_label"]].drop_duplicates()
    counts = unique_videos.groupby("posture_label")["video_id"].nunique()
    print("\nDistinct videos per class:")
    print(counts)
    if counts.min() < 2:
        raise ValueError("Need at least 2 separate videos per class for grouped cross-validation.")

    X = features_df[FEATURES].replace([np.inf, -np.inf], np.nan)
    y = features_df["posture_label"].astype(str)
    groups = features_df["video_id"].astype(str)
    cv = StratifiedGroupKFold(
        n_splits=min(N_SPLITS, int(counts.min())),
        shuffle=True, random_state=RANDOM_STATE
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=RANDOM_STATE, min_samples_leaf=3
        )),
    ])

    predictions, scores = [], []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        if set(y_train) != set(classes):
            warnings.warn(f"Fold {fold} training data does not contain all classes.")
        model.fit(X.iloc[train_idx], y_train)
        pred = model.predict(X.iloc[test_idx])
        scores.append({
            "fold": fold,
            "accuracy": accuracy_score(y_test, pred),
            "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
        })
        predictions.append(pd.DataFrame({
            "video_id": groups.iloc[test_idx].to_numpy(),
            "true_label": y_test.to_numpy(),
            "predicted_label": pred,
            "fold": fold,
        }, index=X.iloc[test_idx].index))
        print(f"Fold {fold}: accuracy={scores[-1]['accuracy']:.3f}, macro-F1={scores[-1]['macro_f1']:.3f}")

    oof = pd.concat(predictions).sort_index()
    score_df = pd.DataFrame(scores)
    score_df.to_csv(OUTPUT_DIR / "cross_validation_scores.csv", index=False)
    oof.to_csv(OUTPUT_DIR / "out_of_fold_frame_predictions.csv", index=False)
    print("\nOut-of-fold frame-level report (preliminary):")
    print(classification_report(oof["true_label"], oof["predicted_label"], labels=classes, zero_division=0))
    print("Confusion matrix; rows=true, columns=predicted; class order:", classes)
    print(confusion_matrix(oof["true_label"], oof["predicted_label"], labels=classes))

    # Majority vote to produce one prediction for each held-out video.
    video_rows = []
    for video_id, group in oof.groupby("video_id"):
        true_label = group["true_label"].mode().iloc[0]
        pred_label = group["predicted_label"].mode().iloc[0]
        video_rows.append({
            "video_id": video_id, "true_label": true_label,
            "predicted_label": pred_label, "frames": len(group),
            "correct": true_label == pred_label,
        })
    video_results = pd.DataFrame(video_rows)
    video_results.to_csv(OUTPUT_DIR / "video_level_predictions.csv", index=False)
    print("\nVideo-level results:")
    print(video_results.to_string(index=False))
    print("Video-level accuracy:", round(video_results["correct"].mean(), 3))

    # Fit final deployment model on all current data; do not use its training score as test score.
    model.fit(X, y)
    model_path = OUTPUT_DIR / "side_posture_random_forest.joblib"
    joblib.dump({"model": model, "feature_columns": FEATURES, "classes": classes}, model_path)
    print(f"\nSaved final model: {model_path}")
    return model

def main():
    raw = extract_videos()
    features = make_features(raw)
    train_classifier(features)
    print("\nFinished. Outputs are in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
