
"""Transparent, project-defined heuristic and LSTM risk scoring."""

import pandas as pd


RISK_WEIGHTS = {
    "anomaly": 45,
    "lstm": 10,
    "offhours": 15,
    "devices": 10,
    "logins": 10,
    "files": 10,
}


def _level(score: float) -> str:
    if score >= 75:
        return "Critical"
    if score >= 50:
        return "High"
    if score >= 25:
        return "Medium"
    return "Low"


def add_risk_scores(scored: pd.DataFrame) -> pd.DataFrame:
    result = scored.copy()

    anomaly_min = result["anomaly_score"].min()
    anomaly_max = result["anomaly_score"].max()

    if anomaly_max == anomaly_min:
        result["anomaly_component"] = 0.0
    else:
        result["anomaly_component"] = (
            (result["anomaly_score"] - anomaly_min)
            / (anomaly_max - anomaly_min)
        ).clip(0, 1)

    thresholds = {
        "devices": result["n_distinct_pcs"].quantile(0.90),
        "logins": result["logon_count"].quantile(0.90),
        "files": result["file_events"].quantile(0.90),
    }

    # LSTM contributes only when inference produced a valid score.
    if "lstm_score" in result.columns:
        lstm_score = pd.to_numeric(result["lstm_score"], errors="coerce")
        result["lstm_component"] = lstm_score.clip(0, 1).fillna(0)
    else:
        result["lstm_component"] = 0.0

    result["risk_score"] = (
        result["anomaly_component"] * RISK_WEIGHTS["anomaly"]
        + result["lstm_component"] * RISK_WEIGHTS["lstm"]
        + result["offhours_logon"].clip(0, 1)
        * RISK_WEIGHTS["offhours"]
        + (result["n_distinct_pcs"] > thresholds["devices"]).astype(int)
        * RISK_WEIGHTS["devices"]
        + (result["logon_count"] > thresholds["logins"]).astype(int)
        * RISK_WEIGHTS["logins"]
        + (result["file_events"] > thresholds["files"]).astype(int)
        * RISK_WEIGHTS["files"]
    ).clip(0, 100).round(2)

    result["risk_level"] = result["risk_score"].map(_level)
    result["explanation"] = result.apply(_explain, axis=1)
    result["alert_required"] = (result["risk_score"] >= 50).astype(int)

    return result


def _explain(row: pd.Series) -> str:
    reasons = []

    if row["anomaly_component"] >= 0.6:
        reasons.append("the activity was anomalous")

    if row.get("lstm_component", 0) >= 0.6:
        reasons.append("the LSTM model detected an unusual activity pattern")

    if row["offhours_logon"]:
        reasons.append("logon activity occurred outside typical hours")

    if row["n_distinct_pcs"] > 1:
        reasons.append(f"activity involved {int(row['n_distinct_pcs'])} computers")

    if row["file_events"] > 0:
        reasons.append(f"{int(row['file_events'])} file events were recorded")

    if not reasons:
        return "No strong contextual indicators were observed in this user-day."

    return f"{row['risk_level']} risk because " + ", ".join(reasons) + "."


def simulation_risk(
    profile: dict,
    event_type: str,
    event_hour: int,
    resource: str | None,
    known_resource: bool,
    known_device: bool,
    recent_suspicious: int,
) -> tuple[float, str, str]:
    """Score one simulated event using baseline deviation and repeated evidence."""
    points = []

    if event_type == "LOGIN" and (event_hour < 6 or event_hour >= 21):
        points.append((22, "login occurred outside the user's normal activity window"))

    if event_type in {"FILE_ACCESS", "FILE_DOWNLOAD"} and resource and not known_resource:
        points.append((18, "the resource is unusual compared with previous activity"))

    if event_type == "FILE_DOWNLOAD":
        points.append((8, "a file download was recorded"))

    if event_type in {"DEVICE_CONNECT", "DEVICE_DISCONNECT"} and not known_device:
        points.append((20, "a device not previously observed for this profile was used"))

    if event_type == "LOGIN" and profile.get("login_count", 0) == 0:
        points.append((4, "this is the first observed login for the simulation profile"))

    if recent_suspicious >= 2:
        points.append((min(25, recent_suspicious * 5), "multiple unusual events occurred recently"))

    base = min(100, sum(value for value, _ in points))

    if not points:
        return base, "Low", "The activity is consistent with the observed simulation baseline."

    level = _level(base)
    return base, level, f"{level} risk because " + ", ".join(
        reason for _, reason in points
    ) + "."
