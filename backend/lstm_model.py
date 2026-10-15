
"""Independent LSTM model for SentinelAI employee-day activity sequences."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

MODEL_PATH = OUTPUT_DIR / "sentinelai_lstm.keras"
SCALER_PATH = OUTPUT_DIR / "sentinelai_lstm_scaler.npz"
METRICS_PATH = OUTPUT_DIR / "sentinelai_lstm_metrics.json"

SEQUENCE_DAYS = 14
RANDOM_SEED = 42

FEATURE_COLUMNS = [
    "logon_count",
    "logoff_count",
    "first_logon_hour",
    "last_logon_hour",
    "n_distinct_pcs",
    "offhours_logon",
    "device_connects",
    "device_disconnects",
    "device_distinct_pcs",
    "file_events",
    "distinct_files",
]


def build_daily_data():
    """Build employee-day features using SentinelAI's existing preprocessing."""
    from backend.preprocessing import build_features

    features = build_features(DATA_DIR)

    missing = set(FEATURE_COLUMNS + [
        "user", "day", "is_malicious_user_day"
    ]) - set(features.columns)

    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    features["day"] = pd.to_datetime(features["day"])
    features = features.sort_values(["user", "day"]).reset_index(drop=True)

    return features


def make_sequences(features):
    """
    Create 14-calendar-day windows for each employee.
    Missing activity days are zero-filled. Their labels are unknown,
    so windows ending on those days are not used as targets.
    """
    X, y, end_dates, users = [], [], [], []

    first_day = features["day"].min()
    last_day = features["day"].max()
    calendar = pd.date_range(first_day, last_day, freq="D")

    for user, group in features.groupby("user", sort=False):
        group = group.set_index("day").sort_index()

        # Keep labels only for dates actually present in the source features.
        labels = group["is_malicious_user_day"].reindex(calendar)
        values = group[FEATURE_COLUMNS].reindex(calendar).fillna(0)

        array = values.to_numpy(dtype=np.float32)
        label_array = labels.to_numpy()

        for end in range(SEQUENCE_DAYS - 1, len(calendar)):
            # Only predict a date that exists in the original feature data.
            if pd.isna(label_array[end]):
                continue

            X.append(array[end - SEQUENCE_DAYS + 1:end + 1])
            y.append(int(label_array[end]))
            end_dates.append(calendar[end])
            users.append(user)

    if not X:
        raise ValueError("No sequences were created. Check the dataset.")

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int32),
        np.asarray(end_dates),
        np.asarray(users),
    )


def build_model():
    model = Sequential([
        Input(shape=(SEQUENCE_DAYS, len(FEATURE_COLUMNS))),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(
                name="pr_auc", curve="PR"
            ),
        ],
    )

    return model


def train_and_evaluate():
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building employee-day features...")
    features = build_daily_data()

    print("Creating 14-day sequences...")
    X, y, dates, users = make_sequences(features)

    print(f"Sequences: {len(X):,}")
    print(f"Positive labels: {int(y.sum()):,}")
    print(f"Negative labels: {int((y == 0).sum()):,}")

    # Split by date, not randomly, to reduce temporal leakage.
    unique_dates = np.sort(np.unique(dates))
    if len(unique_dates) < 10:
        raise ValueError("Not enough dates for a chronological split.")

    train_cutoff = unique_dates[int(len(unique_dates) * 0.70)]
    val_cutoff = unique_dates[int(len(unique_dates) * 0.85)]

    train_mask = dates < train_cutoff
    val_mask = (dates >= train_cutoff) & (dates < val_cutoff)
    test_mask = dates >= val_cutoff

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"Training cutoff: before {train_cutoff}")
    print(f"Validation period: {train_cutoff} to {val_cutoff}")
    print(f"Test period: {val_cutoff} onward")
    print(
        f"Train/validation/test: "
        f"{len(X_train):,}/{len(X_val):,}/{len(X_test):,}"
    )

    if len(np.unique(y_train)) < 2:
        raise ValueError(
            "Training data has only one label class. "
            "A binary LSTM cannot be trained reliably."
        )

    # Fit the scaler on training sequences only.
    scaler = StandardScaler()
    train_2d = X_train.reshape(-1, len(FEATURE_COLUMNS))
    scaler.fit(train_2d)

    def scale_sequences(data):
        shape = data.shape
        scaled = scaler.transform(
            data.reshape(-1, shape[-1])
        )
        return scaled.reshape(shape).astype(np.float32)

    X_train = scale_sequences(X_train)
    X_val = scale_sequences(X_val)
    X_test = scale_sequences(X_test)

    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weights = {
        int(label): float(weight)
        for label, weight in zip(classes, weights)
    }
    print("Class weights:", class_weights)

    model = build_model()

    early_stopping = EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=3,
        restore_best_weights=True,
    )

    print("Training LSTM...")
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=256,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=1,
    )

    # Choose a threshold on validation data, not the test data.
    val_probabilities = model.predict(
        X_val, batch_size=512, verbose=0
    ).ravel()

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in np.arange(0.05, 0.96, 0.05):
        predicted = (val_probabilities >= threshold).astype(int)
        tp = int(((predicted == 1) & (y_val == 1)).sum())
        fp = int(((predicted == 1) & (y_val == 0)).sum())
        fn = int(((predicted == 0) & (y_val == 1)).sum())

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    print(f"Validation-selected threshold: {best_threshold:.2f}")
    print(f"Validation F1 at selected threshold: {best_f1:.4f}")

    print("Evaluating on untouched test period...")
    test_probabilities = model.predict(
        X_test, batch_size=512, verbose=0
    ).ravel()
    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    print("\nConfusion matrix [ [TN, FP], [FN, TP] ]:")
    matrix = confusion_matrix(
        y_test, test_predictions, labels=[0, 1]
    )
    print(matrix)

    print("\nClassification report:")
    report = classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["Normal", "Suspicious"],
        zero_division=0,
        output_dict=True,
    )
    print(classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["Normal", "Suspicious"],
        zero_division=0,
    ))

    metrics = {
        "sequence_days": SEQUENCE_DAYS,
        "feature_columns": FEATURE_COLUMNS,
        "train_cutoff_exclusive": str(train_cutoff),
        "validation_cutoff_exclusive": str(val_cutoff),
        "selected_threshold": best_threshold,
        "validation_f1_at_selected_threshold": best_f1,
        "test_sequences": int(len(y_test)),
        "test_positive_labels": int(y_test.sum()),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }

    if len(np.unique(y_test)) == 2:
        metrics["test_average_precision"] = float(
            average_precision_score(y_test, test_probabilities)
        )
        metrics["test_roc_auc"] = float(
            roc_auc_score(y_test, test_probabilities)
        )
        print(
            "Test average precision (PR-AUC summary):",
            f"{metrics['test_average_precision']:.4f}",
        )
        print(f"Test ROC-AUC: {metrics['test_roc_auc']:.4f}")
    else:
        print(
            "ROC-AUC and average precision were skipped: "
            "the test period contains only one label class."
        )

    model.save(MODEL_PATH)
    if scaler.mean_ is None or scaler.scale_ is None:
        raise RuntimeError("Scaler was not fitted successfully.")

    np.savez(
        SCALER_PATH,
        mean=scaler.mean_,
        scale=scaler.scale_,
    )
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\nSaved LSTM model:", MODEL_PATH)
    print("Saved scaler:", SCALER_PATH)
    print("Saved evaluation:", METRICS_PATH)
    print(
        "\nThis is an independent experimental model. "
        "The existing Isolation Forest was not modified."
    )


if __name__ == "__main__":
    train_and_evaluate()
