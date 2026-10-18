
"""Inference helpers for SentinelAI's trained LSTM model."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .lstm_model import (
    DATA_DIR,
    FEATURE_COLUMNS,
    MODEL_PATH,
    SCALER_PATH,
    SEQUENCE_DAYS,
)
from .preprocessing import build_features


ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = ROOT / "outputs" / "sentinelai_lstm_metrics.json"


def score_lstm(features: pd.DataFrame) -> pd.DataFrame:
    """
    Return LSTM scores for employee-days.

    Output columns: user, day, lstm_score, lstm_flag.
    Missing calendar days are zero-filled. No insider labels are used.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"LSTM model not found: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"LSTM scaler not found: {SCALER_PATH}")

    required = {"user", "day", *FEATURE_COLUMNS}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Missing LSTM input columns: {sorted(missing)}")

    data = features.copy()
    data["day"] = pd.to_datetime(data["day"]).dt.normalize()
    data = data.sort_values(["user", "day"]).reset_index(drop=True)

    with np.load(SCALER_PATH) as saved_scaler:
        mean = saved_scaler["mean"].astype(np.float32)
        scale = saved_scaler["scale"].astype(np.float32)

    if len(mean) != len(FEATURE_COLUMNS) or len(scale) != len(FEATURE_COLUMNS):
        raise ValueError("Saved LSTM scaler does not match the feature columns.")
    scale[scale == 0] = 1.0

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        metrics = json.load(file)
    threshold = float(metrics["selected_threshold"])

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    first_day = data["day"].min()
    last_day = data["day"].max()
    calendar = pd.date_range(first_day, last_day, freq="D")

    output_users = []
    output_days = []
    output_scores = []

    for user, group in data.groupby("user", sort=False):
        group = group.drop_duplicates("day", keep="last").set_index("day")
        daily = group[FEATURE_COLUMNS].reindex(calendar).fillna(0)
        values = daily.to_numpy(dtype=np.float32)

        if len(values) < SEQUENCE_DAYS:
            continue

        scaled = (values - mean) / scale
        windows = np.stack([
            scaled[end - SEQUENCE_DAYS + 1:end + 1]
            for end in range(SEQUENCE_DAYS - 1, len(calendar))
        ]).astype(np.float32)

        probabilities = model.predict(windows, batch_size=512, verbose=0).ravel()

        # Only return dates that exist in the original features.
        observed_days = set(group.index)
        for end, probability in enumerate(probabilities, start=SEQUENCE_DAYS - 1):
            day = calendar[end]
            if day not in observed_days:
                continue
            output_users.append(user)
            output_days.append(day)
            output_scores.append(float(probability))

    results = pd.DataFrame({
        "user": output_users,
        "day": output_days,
        "lstm_score": output_scores,
    })
    results["lstm_flag"] = (results["lstm_score"] >= threshold).astype(int)
    return results


if __name__ == "__main__":
    print("Building employee-day features...")
    daily_features = build_features(DATA_DIR)
    print("Scoring with the saved LSTM...")
    scores = score_lstm(daily_features)
    print(f"Scored employee-days: {len(scores):,}")
    print(f"LSTM flags: {int(scores['lstm_flag'].sum()):,}")
    print(scores.head().to_string(index=False))
