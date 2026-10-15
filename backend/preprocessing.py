"""Load CERT logs and build reusable user-day behavioral features."""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
	"logon.csv": {"date", "user", "pc", "activity"},
	"device.csv": {"date", "user", "pc", "activity"},
	"file.csv": {"date", "user", "pc", "filename"},
	"insiders.csv": {"user", "start", "end"},
}


def _read_csv(data_dir: Path, filename: str) -> pd.DataFrame:
	path = data_dir / filename
	if not path.exists():
		raise FileNotFoundError(f"Required dataset file was not found: {path}")
	frame = pd.read_csv(path, low_memory=False)
	missing = REQUIRED_COLUMNS[filename] - set(frame.columns)
	if missing:
		raise ValueError(f"{filename} is missing required columns: {sorted(missing)}")
	return frame


def _event_features(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
	events = frame.copy()
	events["date"] = pd.to_datetime(events["date"], errors="coerce")
	events = events.dropna(subset=["date", "user"])
	events["user"] = events["user"].astype(str).str.strip()
	events["day"] = events["date"].dt.strftime("%Y-%m-%d")
	events["hour"] = events["date"].dt.hour

	grouped = events.groupby(["user", "day"], as_index=False)
	if prefix == "logon":
		result = grouped.agg(
			logon_count=("activity", lambda values: (values == "Logon").sum()),
			logoff_count=("activity", lambda values: (values == "Logoff").sum()),
			first_logon_hour=("hour", "min"),
			last_logon_hour=("hour", "max"),
			n_distinct_pcs=("pc", "nunique"),
		)
		hours = events.groupby(["user", "day"])["hour"].agg(["min", "max"]).reset_index()
		result = result.merge(hours, on=["user", "day"])
		result["offhours_logon"] = ((result["min"] < 6) | (result["max"] > 20)).astype(int)
		return result.drop(columns=["min", "max"])

	if prefix == "device":
		return grouped.agg(
			device_connects=("activity", lambda values: (values == "Connect").sum()),
			device_disconnects=("activity", lambda values: (values == "Disconnect").sum()),
			device_distinct_pcs=("pc", "nunique"),
		)

	return grouped.agg(
		file_events=("filename", "size"),
		distinct_files=("filename", "nunique"),
	)


def _label_user_days(features: pd.DataFrame, insiders: pd.DataFrame) -> pd.Series:
	intervals = insiders.copy()
	intervals["start"] = pd.to_datetime(intervals["start"], errors="coerce")
	intervals["end"] = pd.to_datetime(intervals["end"], errors="coerce")
	intervals = intervals.dropna(subset=["user", "start", "end"])
	interval_map: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
	for row in intervals.itertuples(index=False):
		interval_map.setdefault(str(row.user), []).append((row.start, row.end))

	def is_inside(row: pd.Series) -> int:
		day_start = pd.Timestamp(row["day"])
		day_end = day_start + pd.Timedelta(days=1)
		return int(any(start < day_end and end >= day_start for start, end in interval_map.get(str(row["user"]), [])))

	return features.apply(is_inside, axis=1)


def build_features(data_dir: str | Path = "data") -> pd.DataFrame:
	"""Return one row per user-day, with optional file activity and labels."""
	data_path = Path(data_dir)
	logon = _read_csv(data_path, "logon.csv")
	device = _read_csv(data_path, "device.csv")
	file_events = _read_csv(data_path, "file.csv")
	insiders = _read_csv(data_path, "insiders.csv")

	features = _event_features(logon, "logon")
	for frame, prefix in ((device, "device"), (file_events, "file")):
		features = features.merge(_event_features(frame, prefix), on=["user", "day"], how="outer")

	numeric_columns = [column for column in features.columns if column not in {"user", "day"}]
	features[numeric_columns] = features[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
	features["is_malicious_user_day"] = _label_user_days(features, insiders)
	return features.sort_values(["day", "user"]).reset_index(drop=True)


def feature_columns(features: pd.DataFrame) -> list[str]:
	excluded = {"user", "day", "is_malicious_user_day", "anomaly_score", "is_flagged"}
	return [column for column in features.columns if column not in excluded and pd.api.types.is_numeric_dtype(features[column])]
