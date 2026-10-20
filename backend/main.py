"""FastAPI application for the SentinelAI dataset-analysis workflow."""

from datetime import datetime, timezone
from pathlib import Path
import secrets
import json
import sqlite3
import threading
import uuid

import pandas as pd
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import database
from .model import DEFAULT_ARTIFACT, score_with_saved_model, train_model
from .preprocessing import build_features
from .risk_engine import add_risk_scores, simulation_risk
from .lstm_inference import METRICS_PATH, score_lstm
from .lstm_model import MODEL_PATH, SCALER_PATH
from .device_monitor import WindowsDeviceMonitor


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = ROOT / "outputs" / "sentinelai.db"
ARTIFACT_PATH = ROOT / DEFAULT_ARTIFACT
app = FastAPI(title="SentinelAI API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
SESSIONS: dict[str, dict] = {}
DEVICE_MONITOR = None
ANALYSIS_JOBS: dict[str, dict] = {}
ANALYSIS_LOCK = threading.Lock()
app.mount("/app", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")


@app.get("/")
def frontend() -> FileResponse:
	return FileResponse(ROOT / "frontend" / "index.html")


class AlertUpdate(BaseModel):
	status: str


class PolicyUpdate(BaseModel):
	confidential_access: bool | None = None
	downloads: bool | None = None
	usb_access: bool | None = None
	monitoring_state: str | None = None


class LoginRequest(BaseModel):
	user_id: str
	password: str


class CreateEmployeeRequest(BaseModel):
	user_id: str
	display_name: str
	password: str


class EventRequest(BaseModel):
	event_type: str
	resource: str | None = None
	device_id: str | None = None
	demo_timestamp: str | None = None


def _now() -> datetime:
	return datetime.now(timezone.utc)


def _session(authorization: str | None = Header(default=None)) -> dict:
	if not authorization or not authorization.startswith("Bearer ") or authorization[7:] not in SESSIONS:
		raise HTTPException(status_code=401, detail="Login required")
	return SESSIONS[authorization[7:]]


def _employee(session: dict = Depends(_session)) -> dict:
	if session["role"] != "employee":
		raise HTTPException(status_code=403, detail="Employee access required")
	return session


def _admin(session: dict = Depends(_session)) -> dict:
	if session["role"] != "admin":
		raise HTTPException(status_code=403, detail="Administrator access required")
	return session


def _timestamp(value: str | None) -> datetime:
	if not value:
		return _now()
	try:
		parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
		return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
	except ValueError as exc:
		raise HTTPException(status_code=400, detail="demo_timestamp must be ISO formatted") from exc


def _features() -> pd.DataFrame:
	return build_features(DATA_DIR)


def _live_alerts() -> list[dict]:
	"""Return one current live alert per employee and calendar day."""
	return database.rows("""
		WITH ranked AS (
			SELECT alerts.*, ROW_NUMBER() OVER (
				PARTITION BY user_id, date(substr(day, 1, 10))
				ORDER BY risk_score DESC, id DESC
			) AS row_number
			FROM alerts
			WHERE source <> 'cert'
		)
		SELECT id, user_id, day, risk_score, risk_level, anomaly_score,
			explanation, status, source
		FROM ranked
		WHERE row_number = 1
		ORDER BY risk_score DESC, day DESC
	""", db_path=DB_PATH)


def _live_risk_history(user_id: str | None = None) -> list[dict]:
	query = """
		SELECT h.* FROM risk_history h
		JOIN activities a ON a.id = h.event_id
		WHERE a.source IN ('application', 'local_device_monitor', 'simulation')
	"""
	params = (user_id,) if user_id else ()
	if user_id:
		query += " AND h.user_id = ?"
	query += " ORDER BY h.timestamp"
	return database.rows(query, params, DB_PATH)


def _run_analysis_job(job_id: str) -> None:
    try:
        with ANALYSIS_LOCK:
            ANALYSIS_JOBS[job_id]["status"] = "running"

        features = _features()

        if ARTIFACT_PATH.exists():
            scored = score_with_saved_model(features, ARTIFACT_PATH)
            metrics = None
        else:
            scored, metrics = train_model(
                features,
                artifact_path=ARTIFACT_PATH,
            )

        lstm_scores = score_lstm(features)

        scored["day"] = pd.to_datetime(scored["day"]).dt.strftime("%Y-%m-%d")
        lstm_scores["day"] = pd.to_datetime(lstm_scores["day"]).dt.strftime("%Y-%m-%d")

        result = scored.merge(
            lstm_scores,
            on=["user", "day"],
            how="left",
        )
        result["lstm_available"] = result["lstm_score"].notna()
        result["lstm_flag"] = result["lstm_flag"].fillna(0).astype(int)

        result = add_risk_scores(result)
        database.replace_analysis(result, DB_PATH)

        with ANALYSIS_LOCK:
            ANALYSIS_JOBS[job_id].update({
                "status": "completed",
                "observations": len(result),
                "alerts": int(result["alert_required"].sum()),
                "metrics": metrics,
            })

    except (FileNotFoundError, ValueError) as exc:
        with ANALYSIS_LOCK:
            ANALYSIS_JOBS[job_id].update({
                "status": "failed",
                "error": str(exc),
            })
    except Exception:
        with ANALYSIS_LOCK:
            ANALYSIS_JOBS[job_id].update({
                "status": "failed",
                "error": "Analysis failed unexpectedly.",
            })



def _active_employee():
	return next((session["user_id"] for session in SESSIONS.values() if session.get("role") == "employee"), None)


def _device_event(user_id, event_type, device):
	try:
		_record_event({"user_id": user_id}, EventRequest(event_type=event_type, resource=device.get("device_id"), device_id=device.get("device_id")))
		item = database.rows("SELECT id FROM activities WHERE user_id = ? ORDER BY id DESC LIMIT 1", (user_id,), DB_PATH)
		if item:
			conn = database.connection(DB_PATH)
			try:
				conn.execute("UPDATE activities SET metadata_json = ?, source = 'local_device_monitor' WHERE id = ?", (json.dumps(device), item[0]["id"]))
				conn.commit()
			finally:
				conn.close()
	except Exception:
		return


@app.on_event("startup")
def start_services():
	global DEVICE_MONITOR
	database.connection(DB_PATH).close()
	threading.Thread(target=_initialize_model, name="sentinel-model-loader", daemon=True).start()
	DEVICE_MONITOR = WindowsDeviceMonitor(_device_event, _active_employee)
	DEVICE_MONITOR.start()


def _initialize_model():
	try:
		features = _features()
		if not database.rows("SELECT 1 FROM results LIMIT 1", db_path=DB_PATH) and ARTIFACT_PATH.exists():
			database.replace_analysis(add_risk_scores(score_with_saved_model(features, ARTIFACT_PATH)), DB_PATH)
	except (FileNotFoundError, ValueError):
		pass


@app.on_event("shutdown")
def stop_services():
	if DEVICE_MONITOR:
		DEVICE_MONITOR.stop()


@app.get("/api/health")
def health() -> dict:
	return {"status": "ok"}


@app.post("/api/auth/login")
def login(request: LoginRequest) -> dict:
	user = database.authenticate(request.user_id, request.password, DB_PATH)
	if not user:
		raise HTTPException(status_code=401, detail="Invalid credentials")
	token = secrets.token_urlsafe(32)
	SESSIONS[token] = user
	_record_event(user, EventRequest(event_type="LOGIN"))
	return {"token": token, "user": user}


@app.post("/api/auth/logout")
def logout(authorization: str | None = Header(default=None)) -> dict:
	if authorization and authorization.startswith("Bearer "):
		session = SESSIONS.get(authorization[7:])
		if session and session.get("role") == "employee":
			_record_event(session, EventRequest(event_type="LOGOUT"))
		SESSIONS.pop(authorization[7:], None)
	return {"logged_out": True}


@app.get("/api/auth/me")
def me(session: dict = Depends(_session)) -> dict:
	return session


@app.get("/api/employee/profile")
def employee_profile(session: dict = Depends(_employee)) -> dict:
	profile = database.rows("SELECT * FROM behavior_profiles WHERE user_id = ?", (session["user_id"],), DB_PATH)[0]
	policy = database.rows("SELECT * FROM restrictions WHERE user_id = ?", (session["user_id"],), DB_PATH)[0]
	return {"user": session, "profile": profile, "policy": policy}


@app.get("/api/employee/files")
def employee_files(_: dict = Depends(_employee)) -> list[dict]:
	return database.rows("SELECT file_id, name, classification FROM simulation_files ORDER BY classification, name", db_path=DB_PATH)


@app.get("/api/employee/activity")
def employee_activity(session: dict = Depends(_employee)) -> list[dict]:
	return database.rows("SELECT id, timestamp, event_type, resource, metadata_json, source FROM activities WHERE user_id = ? ORDER BY timestamp DESC LIMIT 100", (session["user_id"],), DB_PATH)


@app.get("/api/employee/devices")
def employee_devices(session: dict = Depends(_employee)) -> list[dict]:
	devices = database.rows("SELECT device_id, label FROM simulation_devices ORDER BY label", db_path=DB_PATH)
	for device in devices:
		latest = database.rows("SELECT event_type, timestamp FROM activities WHERE user_id = ? AND resource = ? AND event_type IN ('DEVICE_CONNECT', 'DEVICE_DISCONNECT') ORDER BY timestamp DESC LIMIT 1", (session["user_id"], device["device_id"]), DB_PATH)
		device["connected"] = bool(latest and latest[0]["event_type"] == "DEVICE_CONNECT")
	return devices


def _record_event(session: dict, request: EventRequest) -> dict:
	allowed = {"LOGIN", "LOGOUT", "FILE_ACCESS", "FILE_DOWNLOAD", "FILE_UPLOAD", "DEVICE_CONNECT", "DEVICE_DISCONNECT"}
	if request.event_type not in allowed:
		raise HTTPException(status_code=400, detail="Unsupported event type")
	event_time = _timestamp(request.demo_timestamp)
	user_id = session["user_id"]
	policy = database.rows("SELECT * FROM restrictions WHERE user_id = ?", (user_id,), DB_PATH)[0]
	file_info = database.rows("SELECT classification FROM simulation_files WHERE name = ?", (request.resource,), DB_PATH) if request.resource else []
	if request.event_type == "FILE_ACCESS" and (policy["file_access"] or file_info and file_info[0]["classification"] == "CONFIDENTIAL" and policy["confidential_access"]):
		database.add_activity(user_id, "POLICY_DENIED", event_time.isoformat(), request.resource, {"attempted_event": request.event_type}, source="policy_engine", db_path=DB_PATH)
		raise HTTPException(status_code=403, detail="Access to this resource is currently restricted by your organization's security policy.")
	if request.event_type == "FILE_DOWNLOAD" and (policy["downloads"] or file_info and file_info[0]["classification"] == "CONFIDENTIAL" and policy["confidential_access"]):
		database.add_activity(user_id, "POLICY_DENIED", event_time.isoformat(), request.resource, {"attempted_event": request.event_type}, source="policy_engine", db_path=DB_PATH)
		raise HTTPException(status_code=403, detail="Downloads for this resource are restricted.")
	if request.event_type in {"DEVICE_CONNECT", "DEVICE_DISCONNECT"} and policy["usb_access"]:
		database.add_activity(user_id, "POLICY_DENIED", event_time.isoformat(), request.resource, {"attempted_event": request.event_type}, source="policy_engine", db_path=DB_PATH)
		raise HTTPException(status_code=403, detail="Device access is currently restricted by your organization's security policy.")
	known_resource = bool(database.rows("SELECT 1 FROM activities WHERE user_id = ? AND resource = ? LIMIT 1", (user_id, request.resource), DB_PATH)) if request.resource else True
	known_device = bool(request.device_id and database.rows("SELECT 1 FROM activities WHERE user_id = ? AND resource = ? LIMIT 1", (user_id, request.device_id), DB_PATH))
	profile = database.rows("SELECT * FROM behavior_profiles WHERE user_id = ?", (user_id,), DB_PATH)[0]
	recent = database.rows("SELECT COUNT(*) AS count FROM activities WHERE user_id = ? AND timestamp >= datetime(?, '-24 hours') AND event_type IN ('FILE_DOWNLOAD', 'DEVICE_CONNECT')", (user_id, event_time.isoformat()), DB_PATH)[0]["count"]
	risk_score, risk_level, explanation = simulation_risk(profile, request.event_type, event_time.hour, request.resource, known_resource, known_device, recent)
	resource = request.resource or request.device_id
	event_id = database.add_activity(user_id, request.event_type, event_time.isoformat(), resource, {"risk_level": risk_level}, source="application", db_path=DB_PATH)
	database.update_profile(user_id, request.event_type, event_time.hour, request.device_id, DB_PATH)
	conn = database.connection(DB_PATH)
	try:
		conn.execute("INSERT INTO risk_history(user_id, timestamp, event_id, risk_score, risk_level, anomaly_score, explanation) VALUES (?, ?, ?, ?, ?, ?, ?)", (user_id, event_time.isoformat(), event_id, max(0.0, min(100.0, float(risk_score))), risk_level, 0.0, explanation))
		conn.commit()
	finally:
		conn.close()
	if risk_score >= 50:
		conn = database.connection(DB_PATH)
		try:
			conn.execute("INSERT INTO alerts(user_id, day, risk_score, risk_level, anomaly_score, explanation, source) VALUES (?, ?, ?, ?, ?, ?, 'behavioral_monitor') ON CONFLICT(user_id, day) DO UPDATE SET risk_score = excluded.risk_score, risk_level = excluded.risk_level, anomaly_score = excluded.anomaly_score, explanation = excluded.explanation, source = 'behavioral_monitor'", (user_id, event_time.date().isoformat(), max(0.0, min(100.0, float(risk_score))), risk_level, 0.0, explanation))
			conn.commit()
		finally:
			conn.close()
	return {"event_id": event_id, "timestamp": event_time.isoformat(), "message": "Activity recorded in the company workspace.", "restricted": False}


@app.post("/api/employee/events")
def employee_event(request: EventRequest, session: dict = Depends(_employee)) -> dict:
	return _record_event(session, request)


@app.get("/api/admin/dashboard")
def admin_dashboard(_: dict = Depends(_admin)) -> dict:
	activities = database.rows("SELECT COUNT(*) AS count FROM activities", db_path=DB_PATH)[0]["count"]
	events_today = database.rows("SELECT COUNT(*) AS count FROM activities WHERE date(timestamp) = date('now')", db_path=DB_PATH)[0]["count"]
	users = database.rows("SELECT COUNT(*) AS count FROM users WHERE role = 'employee'", db_path=DB_PATH)[0]["count"]
	alerts_data = _live_alerts()
	profiles = database.rows("""
		SELECT u.display_name, r.user_id, r.monitoring_state, r.file_access,
			r.confidential_access, r.downloads, r.usb_access,
			COALESCE(MAX(h.risk_score), 0) AS current_risk
		FROM users u
		JOIN restrictions r ON r.user_id = u.user_id
		LEFT JOIN risk_history h ON h.user_id = u.user_id
		LEFT JOIN activities a ON a.id = h.event_id
		WHERE u.role = 'employee'
			AND (a.source IS NULL OR a.source IN ('application', 'local_device_monitor', 'simulation'))
		GROUP BY r.user_id
	""", db_path=DB_PATH)
	risk_distribution = [{"label": level, "count": sum(item["risk_level"] == level for item in alerts_data)} for level in ("Critical", "High", "Medium", "Low")]
	live_history = _live_risk_history()
	return {"summary": {"total_employees": users, "total_activities": activities, "events_today": events_today, "anomalous_events": sum(item["risk_level"] in {"High", "Critical"} for item in alerts_data), "high_risks": sum(item["risk_level"] in {"High", "Critical"} for item in alerts_data), "critical_risks": sum(item["risk_level"] == "Critical" for item in alerts_data), "active_alerts": sum(item["status"] != "Resolved" for item in alerts_data)}, "users": profiles, "alerts": alerts_data[:30], "activity_types": database.rows("SELECT event_type AS label, COUNT(*) AS count FROM activities WHERE source IN ('application', 'local_device_monitor', 'simulation') GROUP BY event_type ORDER BY count DESC", db_path=DB_PATH), "risk_distribution": risk_distribution, "live_activity": database.rows("SELECT a.*, u.display_name FROM activities a JOIN users u ON u.user_id = a.user_id WHERE a.source IN ('application', 'local_device_monitor', 'simulation') ORDER BY a.timestamp DESC LIMIT 25", db_path=DB_PATH), "risk_trend": [{"user_id": item["user_id"], "timestamp": item["timestamp"], "risk_score": item["risk_score"], "risk_level": item["risk_level"]} for item in live_history[-60:]]}


@app.post("/api/admin/users", status_code=201)
def create_employee(employee: CreateEmployeeRequest, _: dict = Depends(_admin)) -> dict:
	user_id = employee.user_id.strip()
	display_name = employee.display_name.strip()
	password = employee.password

	if not 3 <= len(user_id) <= 32 or not all(char.isalnum() or char in "_-" for char in user_id):
		raise HTTPException(status_code=400, detail="Employee ID must be 3–32 characters and use only letters, numbers, hyphens, or underscores.")
	if not 2 <= len(display_name) <= 80:
		raise HTTPException(status_code=400, detail="Full name must be between 2 and 80 characters.")
	if len(password) < 8 or len(password) > 128:
		raise HTTPException(status_code=400, detail="Password must be between 8 and 128 characters.")

	conn = database.connection(DB_PATH)
	try:
		if conn.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,)).fetchone():
			raise HTTPException(status_code=409, detail="That employee ID already exists.")
		conn.execute(
			"""INSERT INTO users(
				user_id, display_name, role, password_hash, status, is_demo
			) VALUES (?, ?, 'employee', ?, 'active', 0)""",
			(user_id, display_name, database.hash_password(password)),
		)
		conn.execute("INSERT INTO behavior_profiles(user_id) VALUES (?)", (user_id,))
		conn.execute("INSERT INTO restrictions(user_id) VALUES (?)", (user_id,))
		conn.commit()
	except sqlite3.IntegrityError as exc:
		conn.rollback()
		raise HTTPException(status_code=409, detail="That employee ID already exists.") from exc
	finally:
		conn.close()
	return {"created": True, "user": {"user_id": user_id, "display_name": display_name, "role": "employee"}}


@app.get("/api/admin/users")
def admin_users(_: dict = Depends(_admin)) -> list[dict]:
	return database.rows("""
		SELECT u.user_id, u.employee_code, u.email, u.display_name, u.department,
			u.job_title, u.manager_code, u.status, u.is_demo,
			r.monitoring_state, r.file_access,
			COALESCE(MAX(h.risk_score), 0) AS current_risk
		FROM users u
		JOIN restrictions r ON r.user_id = u.user_id
		LEFT JOIN risk_history h ON h.user_id = u.user_id
		LEFT JOIN activities a ON a.id = h.event_id
		WHERE u.role = 'employee'
			AND (a.source IS NULL OR a.source IN ('application', 'local_device_monitor', 'simulation'))
		GROUP BY u.user_id
	""", db_path=DB_PATH)


@app.get("/api/admin/demo-credentials")
def demo_credentials(_: dict = Depends(_admin)) -> list[dict]:
	return [
		{
			"name": display_name,
			"employee_code": employee_code,
			"login_identifier": email,
			"password": f"{display_name.split()[0].lower()}1",
		}
		for employee_code, display_name, _, _, _ in database.DEMO_EMPLOYEES
		for email in [f"{display_name.lower().replace(' ', '.')}@sentinelai.demo"]
	]


@app.get("/api/admin/users/{user_id}")
def admin_user(user_id: str, _: dict = Depends(_admin)) -> dict:
	return {"profile": database.rows("SELECT * FROM behavior_profiles WHERE user_id = ?", (user_id,), DB_PATH), "policy": database.rows("SELECT * FROM restrictions WHERE user_id = ?", (user_id,), DB_PATH), "activity": database.rows("SELECT * FROM activities WHERE user_id = ? AND source IN ('application', 'local_device_monitor', 'simulation') ORDER BY timestamp DESC LIMIT 100", (user_id,), DB_PATH), "risk_history": _live_risk_history(user_id)}


@app.get("/api/admin/lstm/summary")
def admin_lstm_summary(page: int = 1, page_size: int = 25, _: dict = Depends(_admin)) -> dict:
	if page < 1 or page_size < 1 or page_size > 100:
		raise HTTPException(status_code=400, detail="page must be at least 1 and page_size must be between 1 and 100")
	artifacts_available = all(path.exists() for path in (MODEL_PATH, SCALER_PATH, METRICS_PATH))
	metrics = {}
	if METRICS_PATH.exists():
		try:
			metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
		except (OSError, TypeError, ValueError) as exc:
			raise HTTPException(status_code=500, detail="Saved LSTM metrics could not be read.") from exc
	summary = database.rows("""
		SELECT COUNT(*) AS total_scored,
			SUM(CASE WHEN json_extract(features_json, '$.lstm_flag') = 1 THEN 1 ELSE 0 END) AS flagged_count
		FROM results
		WHERE json_extract(features_json, '$.lstm_available') = 1
	""", db_path=DB_PATH)[0]
	total_scored = int(summary["total_scored"] or 0)
	flagged_count = int(summary["flagged_count"] or 0)
	offset = (page - 1) * page_size
	flagged_rows = database.rows("""
		SELECT user_id, day,
			json_extract(features_json, '$.lstm_score') AS lstm_score,
			json_extract(features_json, '$.lstm_flag') AS lstm_flag,
			json_extract(features_json, '$.lstm_available') AS lstm_available
		FROM results
		WHERE json_extract(features_json, '$.lstm_available') = 1
			AND json_extract(features_json, '$.lstm_flag') = 1
		ORDER BY lstm_score DESC, day DESC, user_id
		LIMIT ? OFFSET ?
	""", (page_size, offset), DB_PATH)
	distribution = database.rows("""
		SELECT CASE
			WHEN json_extract(features_json, '$.lstm_score') < 0.25 THEN '0.00-0.24'
			WHEN json_extract(features_json, '$.lstm_score') < 0.50 THEN '0.25-0.49'
			WHEN json_extract(features_json, '$.lstm_score') < 0.75 THEN '0.50-0.74'
			ELSE '0.75-1.00'
		END AS label, COUNT(*) AS count
		FROM results
		WHERE json_extract(features_json, '$.lstm_available') = 1
		GROUP BY label
		ORDER BY label
	""", db_path=DB_PATH)
	return {
		"model": {
			"status": "available" if artifacts_available else "unavailable",
			"artifacts_available": artifacts_available,
			"analysis_available": bool(total_scored),
			"sequence_days": metrics.get("sequence_days"),
			"threshold": metrics.get("selected_threshold"),
		},
		"summary": {
			"total_scored": total_scored,
			"flagged_count": flagged_count,
			"flagged_percentage": round((flagged_count / total_scored) * 100, 2) if total_scored else 0,
		},
		"distribution": distribution,
		"flagged_employee_days": flagged_rows,
		"pagination": {
			"page": page,
			"page_size": page_size,
			"total": flagged_count,
			"total_pages": (flagged_count + page_size - 1) // page_size,
		},
	}


@app.get("/api/admin/alerts")
def admin_alerts(_: dict = Depends(_admin)) -> list[dict]:
	return _live_alerts()


@app.patch("/api/admin/alerts/{alert_id}")
def admin_patch_alert(alert_id: int, update: AlertUpdate, _: dict = Depends(_admin)) -> dict:
	return _patch_alert(alert_id, update, source="live")


@app.post("/api/admin/users/{user_id}/increase-monitoring")
def increase_monitoring(user_id: str, _: dict = Depends(_admin)) -> dict:
	database.set_policy(user_id, monitoring_state="ENHANCED_MONITORING", db_path=DB_PATH)
	return {"monitoring_state": "ENHANCED_MONITORING"}


@app.post("/api/admin/users/{user_id}/restrict-file-access")
def restrict_files(user_id: str, _: dict = Depends(_admin)) -> dict:
	database.set_policy(user_id, file_access=1, monitoring_state="RESTRICTED", db_path=DB_PATH)
	return {"file_access": "restricted", "monitoring_state": "RESTRICTED"}


@app.post("/api/admin/users/{user_id}/policy")
def update_policy(user_id: str, update: PolicyUpdate, _: dict = Depends(_admin)) -> dict:
	values = {key: None if value is None else int(value) for key, value in update.model_dump().items() if key != "monitoring_state"}
	database.set_policy(user_id, **values, monitoring_state=update.monitoring_state, db_path=DB_PATH)
	return database.rows("SELECT * FROM restrictions WHERE user_id = ?", (user_id,), DB_PATH)[0]


@app.post("/api/admin/users/{user_id}/restore-file-access")
def restore_files(user_id: str, _: dict = Depends(_admin)) -> dict:
	database.set_policy(user_id, file_access=0, monitoring_state="NORMAL", db_path=DB_PATH)
	return {"file_access": "restored", "monitoring_state": "NORMAL"}


def _expand_lstm_fields(rows: list[dict]) -> list[dict]:
	for row in rows:
		raw = row.get("features_json")
		try:
			features = json.loads(raw) if isinstance(raw, str) else {}
		except (TypeError, ValueError):
			features = {}

		for key in ("lstm_score", "lstm_flag", "lstm_available"):
			if key in features:
				row[key] = features[key]
	return rows


@app.get("/api/dashboard")
def dashboard(_: dict = Depends(_admin)) -> dict:
	results = _expand_lstm_fields(database.rows("SELECT * FROM results ORDER BY day DESC, risk_score DESC"))
	alerts = database.rows("SELECT * FROM alerts WHERE source = 'cert' ORDER BY risk_score DESC")
	if not results:
		return {"has_data": False, "summary": {}, "risk_distribution": [], "activity_distribution": [], "recent_alerts": []}
	return {"has_data": True, "summary": {"total_users": len({row["user_id"] for row in results}), "anomalies": sum(row["is_flagged"] for row in results), "high_risks": sum(row["risk_level"] in {"High", "Critical"} for row in results), "active_alerts": sum(row["status"] != "Resolved" for row in alerts)}, "risk_distribution": _group_counts(results, "risk_level"), "activity_distribution": _group_counts(results, "is_flagged"), "risk_trend": _trend(results), "recent_alerts": alerts[:20]}


def _group_counts(rows: list[dict], key: str) -> list[dict]:
	counts = {}
	for row in rows:
		label = "Anomalous" if key == "is_flagged" and row[key] else "Normal" if key == "is_flagged" else row[key]
		counts[label] = counts.get(label, 0) + 1
	return [{"label": label, "count": count} for label, count in counts.items()]


def _trend(results: list[dict]) -> list[dict]:
	totals = {}
	for row in results:
		totals.setdefault(row["day"], []).append(row["risk_score"])
	return [{"day": day, "risk": round(sum(scores) / len(scores), 2)} for day, scores in sorted(totals.items())]


@app.get("/api/users")
def users(_: dict = Depends(_admin)) -> list[dict]:
	return database.rows("SELECT user_id, COUNT(*) AS observations, MAX(risk_score) AS max_risk FROM results GROUP BY user_id ORDER BY max_risk DESC")


@app.get("/api/users/{user_id}")
def user(user_id: str, _: dict = Depends(_admin)) -> list[dict]:
	return _expand_lstm_fields(database.rows("SELECT * FROM results WHERE user_id = ? ORDER BY day DESC", (user_id,)))


@app.get("/api/anomalies")
def anomalies(_: dict = Depends(_admin)) -> list[dict]:
	return _expand_lstm_fields(database.rows("SELECT * FROM results WHERE is_flagged = 1 ORDER BY risk_score DESC"))


@app.get("/api/alerts")
def alerts(_: dict = Depends(_admin)) -> list[dict]:
	return database.rows("SELECT * FROM alerts WHERE source = 'cert' ORDER BY risk_score DESC")


@app.get("/api/risk-summary")
def risk_summary(_: dict = Depends(_admin)) -> list[dict]:
	return database.rows("SELECT risk_level, COUNT(*) AS count FROM results GROUP BY risk_level")


@app.patch("/api/alerts/{alert_id}")
def patch_alert(alert_id: int, update: AlertUpdate, _: dict = Depends(_admin)) -> dict:
	return _patch_alert(alert_id, update, source="cert")


def _patch_alert(alert_id: int, update: AlertUpdate, source: str) -> dict:
	try:
		updated = database.update_alert(alert_id, update.status, DB_PATH, source=source)
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	if not updated:
		raise HTTPException(status_code=404, detail="Alert not found")
	return {"updated": True}


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...), _: dict = Depends(_admin)) -> dict:
	accepted = {"logon.csv", "device.csv", "file.csv", "insiders.csv"}
	saved = []
	for upload_file in files:
		if upload_file.filename not in accepted:
			raise HTTPException(status_code=400, detail=f"Unsupported dataset file: {upload_file.filename}")
		path = DATA_DIR / upload_file.filename
		with path.open("wb") as destination:
			while chunk := await upload_file.read(1024 * 1024):
				destination.write(chunk)
		saved.append(upload_file.filename)
	return {"saved": saved}


@app.post("/api/train")
def train(_: dict = Depends(_admin)) -> dict:
	try:
		_, metrics = train_model(_features(), artifact_path=ARTIFACT_PATH)
		return {"trained": True, "metrics": metrics}
	except (FileNotFoundError, ValueError) as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/analyze")
def analyze(_: dict = Depends(_admin)) -> dict:
	with ANALYSIS_LOCK:
		if any(job["status"] in {"queued", "running"} for job in ANALYSIS_JOBS.values()):
			raise HTTPException(status_code=409, detail="A CERT analysis is already running.")
		job_id = uuid.uuid4().hex
		ANALYSIS_JOBS[job_id] = {"status": "queued"}
	threading.Thread(target=_run_analysis_job, args=(job_id,), name=f"sentinel-analysis-{job_id[:8]}", daemon=True).start()
	return {"accepted": True, "job_id": job_id, "status": "queued"}


@app.get("/api/analyze/{job_id}")
def analyze_status(job_id: str, _: dict = Depends(_admin)) -> dict:
	with ANALYSIS_LOCK:
		job = ANALYSIS_JOBS.get(job_id)
		if not job:
			raise HTTPException(status_code=404, detail="Analysis job not found")
		return {"job_id": job_id, **job}
