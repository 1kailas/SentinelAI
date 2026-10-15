"""Unsupervised Isolation Forest training and scoring."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .preprocessing import feature_columns


DEFAULT_ARTIFACT = Path("outputs/sentinelai_model.joblib")


def train_model(
	features: pd.DataFrame,
	n_estimators: int = 200,
	contamination: float = 0.05,
	random_state: int = 42,
	artifact_path: str | Path = DEFAULT_ARTIFACT,
) -> tuple[pd.DataFrame, dict]:
	columns = feature_columns(features)
	if not columns or len(features) < 2:
		raise ValueError("At least two user-day observations with numeric features are required.")
	scaler = StandardScaler()
	matrix = scaler.fit_transform(features[columns].fillna(0))
	model = IsolationForest(
		n_estimators=n_estimators,
		contamination=contamination,
		random_state=random_state,
		n_jobs=-1,
	)
	model.fit(matrix)
	scored = features.copy()
	scored["anomaly_score"] = -model.decision_function(matrix)
	scored["is_flagged"] = (model.predict(matrix) == -1).astype(int)
	metrics = evaluate(scored)
	path = Path(artifact_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump({"model": model, "scaler": scaler, "feature_columns": columns}, path)
	return scored, metrics


def score_with_saved_model(features: pd.DataFrame, artifact_path: str | Path = DEFAULT_ARTIFACT) -> pd.DataFrame:
	artifact = joblib.load(artifact_path)
	columns = artifact["feature_columns"]
	matrix = artifact["scaler"].transform(features[columns].fillna(0))
	scored = features.copy()
	scored["anomaly_score"] = -artifact["model"].decision_function(matrix)
	scored["is_flagged"] = (artifact["model"].predict(matrix) == -1).astype(int)
	return scored


def evaluate(scored: pd.DataFrame) -> dict:
	if "is_malicious_user_day" not in scored or scored["is_malicious_user_day"].nunique() < 2:
		return {"available": False, "reason": "Ground-truth labels do not contain both classes."}
	actual = scored["is_malicious_user_day"]
	predicted = scored["is_flagged"]
	report = classification_report(actual, predicted, output_dict=True, zero_division=0)
	return {
		"available": True,
		"confusion_matrix": confusion_matrix(actual, predicted).tolist(),
		"precision": report["1"]["precision"],
		"recall": report["1"]["recall"],
		"f1": report["1"]["f1-score"],
		"roc_auc": roc_auc_score(actual, scored["anomaly_score"]),
	}
