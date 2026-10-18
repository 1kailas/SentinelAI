"""SQLite persistence for analysis results and alerts."""

import json
import hashlib
import os
import sqlite3
import threading
from pathlib import Path

import pandas as pd


DEFAULT_DB = Path("outputs/sentinelai.db")
ADMIN_USER = ("admin", "Security Administrator", "admin", "admin-demo")
DEMO_EMPLOYEES = [
	("EMP-003", "Adam Sharma", "Engineering", "Software Engineer", "EMP-010"),
	("EMP-004", "Aarav Mehta", "Engineering", "Senior Software Engineer", "EMP-010"),
	("EMP-005", "Diya Kapoor", "Product", "Product Designer", "EMP-011"),
	("EMP-006", "Riya Nair", "Human Resources", "People Operations Specialist", "EMP-012"),
	("EMP-007", "Kabir Singh", "Finance", "Financial Analyst", "EMP-013"),
	("EMP-008", "Ananya Rao", "Engineering", "QA Engineer", "EMP-010"),
	("EMP-009", "Vihaan Patel", "Sales", "Account Executive", "EMP-014"),
	("EMP-010", "Maya Iyer", "Engineering", "Engineering Manager", None),
	("EMP-011", "Rohan Malhotra", "Product", "Product Manager", None),
	("EMP-012", "Neha Joshi", "Human Resources", "HR Manager", None),
	("EMP-013", "Arjun Desai", "Finance", "Finance Manager", None),
	("EMP-014", "Ishaan Verma", "Sales", "Sales Manager", None),
	("EMP-015", "Meera Shah", "Marketing", "Marketing Specialist", "EMP-016"),
	("EMP-016", "Kavya Menon", "Marketing", "Marketing Manager", None),
	("EMP-017", "Ritvik Bansal", "Engineering", "Data Engineer", "EMP-010"),
	("EMP-018", "Saanvi Gupta", "Engineering", "UX Researcher", "EMP-011"),
	("EMP-019", "Aditya Kulkarni", "Operations", "Operations Analyst", "EMP-020"),
	("EMP-020", "Tara Choudhary", "Operations", "Operations Manager", None),
	("EMP-021", "Nikhil Sethi", "IT", "Systems Administrator", "EMP-022"),
	("EMP-022", "Ira Mukherjee", "IT", "IT Manager", None),
	("EMP-023", "Dev Khanna", "Legal", "Compliance Analyst", "EMP-024"),
	("EMP-024", "Aisha Thomas", "Legal", "Compliance Manager", None),
	("EMP-025", "Yash Agarwal", "Customer Success", "Customer Success Specialist", "EMP-026"),
	("EMP-026", "Pooja Reddy", "Customer Success", "Customer Success Manager", None),
	("EMP-027", "Reyansh Bose", "Research", "Research Associate", "EMP-010"),
]
SIMULATION_FILES = [
	("project_report.pdf", "NORMAL"),
	("project_data.csv", "NORMAL"),
	("company_policy.pdf", "NORMAL"),
	("employee_handbook.pdf", "NORMAL"),
	("confidential_financials.pdf", "CONFIDENTIAL"),
	("confidential_strategy.pdf", "CONFIDENTIAL"),
]
INITIALIZE_LOCK = threading.RLock()
INITIALIZED_DATABASES: set[str] = set()


def hash_password(password: str, salt: bytes | None = None) -> str:
	salt = salt or os.urandom(16)
	digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 120_000)
	return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
	try:
		salt_hex, digest_hex = encoded.split("$", 1)
		digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), 120_000)
		return digest.hex() == digest_hex
	except (ValueError, TypeError):
		return False


def connection(db_path: str | Path = DEFAULT_DB) -> sqlite3.Connection:
	path = Path(db_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	conn = sqlite3.connect(path, timeout=30)
	conn.row_factory = sqlite3.Row
	conn.execute("PRAGMA busy_timeout = 30000")
	resolved_path = str(path.resolve())
	with INITIALIZE_LOCK:
		if resolved_path not in INITIALIZED_DATABASES:
			initialize(conn)
			INITIALIZED_DATABASES.add(resolved_path)
	return conn


def initialize(conn: sqlite3.Connection) -> None:
	conn.executescript("""
		CREATE TABLE IF NOT EXISTS users (
			user_id TEXT PRIMARY KEY, display_name TEXT NOT NULL, role TEXT NOT NULL,
			password_hash TEXT NOT NULL, employee_code TEXT UNIQUE, email TEXT UNIQUE,
			department TEXT, job_title TEXT, manager_code TEXT,
			status TEXT NOT NULL DEFAULT 'active', is_demo INTEGER NOT NULL DEFAULT 0,
			created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
		);
		CREATE TABLE IF NOT EXISTS activities (
			id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, timestamp TEXT NOT NULL,
			event_type TEXT NOT NULL, resource TEXT, metadata_json TEXT NOT NULL DEFAULT '{}',
			source TEXT NOT NULL DEFAULT 'simulation'
		);
		CREATE TABLE IF NOT EXISTS behavior_profiles (
			user_id TEXT PRIMARY KEY, typical_login_hour REAL, login_count INTEGER NOT NULL DEFAULT 0,
			file_access_count INTEGER NOT NULL DEFAULT 0, download_count INTEGER NOT NULL DEFAULT 0,
			device_count INTEGER NOT NULL DEFAULT 0, offhours_count INTEGER NOT NULL DEFAULT 0,
			known_devices_json TEXT NOT NULL DEFAULT '[]', updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
		);
		CREATE TABLE IF NOT EXISTS risk_history (
			id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, timestamp TEXT NOT NULL,
			event_id INTEGER, risk_score REAL NOT NULL, risk_level TEXT NOT NULL,
			anomaly_score REAL NOT NULL, explanation TEXT NOT NULL
		);
		CREATE TABLE IF NOT EXISTS restrictions (
			user_id TEXT PRIMARY KEY, file_access INTEGER NOT NULL DEFAULT 0,
			confidential_access INTEGER NOT NULL DEFAULT 0,
			downloads INTEGER NOT NULL DEFAULT 0,
			usb_access INTEGER NOT NULL DEFAULT 0,
			monitoring_state TEXT NOT NULL DEFAULT 'NORMAL', updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
		);
		CREATE TABLE IF NOT EXISTS simulation_files (
			file_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL, classification TEXT NOT NULL
		);
		CREATE TABLE IF NOT EXISTS simulation_devices (
			device_id TEXT PRIMARY KEY, label TEXT NOT NULL
		);
		CREATE TABLE IF NOT EXISTS results (
			id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, day TEXT NOT NULL,
			anomaly_score REAL NOT NULL, is_flagged INTEGER NOT NULL, risk_score REAL NOT NULL,
			risk_level TEXT NOT NULL, explanation TEXT NOT NULL, features_json TEXT NOT NULL,
			UNIQUE(user_id, day)
		);
		CREATE TABLE IF NOT EXISTS alerts (
			id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, day TEXT NOT NULL,
			risk_score REAL NOT NULL, risk_level TEXT NOT NULL, anomaly_score REAL NOT NULL,
			explanation TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'New', source TEXT NOT NULL DEFAULT 'cert',
			UNIQUE(user_id, day)
		);
	""")
	for column, definition in (
		("employee_code", "TEXT"),
		("email", "TEXT"),
		("department", "TEXT"),
		("job_title", "TEXT"),
		("manager_code", "TEXT"),
		("status", "TEXT NOT NULL DEFAULT 'active'"),
		("is_demo", "INTEGER NOT NULL DEFAULT 0"),
	):
		try:
			conn.execute(f"ALTER TABLE users ADD COLUMN {column} {definition}")
		except sqlite3.OperationalError:
			pass
	try:
		conn.execute("ALTER TABLE alerts ADD COLUMN source TEXT NOT NULL DEFAULT 'cert'")
	except sqlite3.OperationalError:
		pass
	for column in ("confidential_access", "downloads", "usb_access"):
		try:
			conn.execute(f"ALTER TABLE restrictions ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0")
		except sqlite3.OperationalError:
			pass
	admin_id, admin_name, admin_role, admin_password = ADMIN_USER
	if conn.execute("SELECT 1 FROM users WHERE user_id = ?", (admin_id,)).fetchone() is None:
		conn.execute("INSERT INTO users(user_id, display_name, role, password_hash) VALUES (?, ?, ?, ?)", (admin_id, admin_name, admin_role, hash_password(admin_password, admin_id.encode()[:16])))
	conn.execute("UPDATE users SET role = 'admin', is_demo = 0, status = 'active' WHERE user_id = ?", (admin_id,))

	legacy_ids = ("employee1", "employee2")
	for user_id in legacy_ids:
		conn.execute("DELETE FROM risk_history WHERE user_id = ?", (user_id,))
		conn.execute("DELETE FROM activities WHERE user_id = ?", (user_id,))
		conn.execute("DELETE FROM alerts WHERE user_id = ? AND source <> 'cert'", (user_id,))
		conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
		conn.execute("DELETE FROM behavior_profiles WHERE user_id = ?", (user_id,))
		conn.execute("DELETE FROM restrictions WHERE user_id = ?", (user_id,))
	for employee_code, display_name, department, job_title, manager_code in DEMO_EMPLOYEES:
		email = f"{display_name.lower().replace(' ', '.')}@sentinelai.demo"
		first_name = display_name.split()[0].lower()
		values = (employee_code, display_name, email, department, job_title, manager_code)
		if conn.execute("SELECT 1 FROM users WHERE user_id = ?", (employee_code,)).fetchone() is None:
			conn.execute(
				"""INSERT INTO users(
					user_id, display_name, role, password_hash, employee_code, email,
					department, job_title, manager_code, status, is_demo
				) VALUES (?, ?, 'employee', ?, ?, ?, ?, ?, ?, 'active', 1)""",
				(employee_code, display_name, hash_password(f"{first_name}1", employee_code.encode()[:16]), values[0], *values[2:]),
			)
		else:
			conn.execute(
				"""UPDATE users SET display_name = ?, role = 'employee', employee_code = ?,
					email = ?, department = ?, job_title = ?, manager_code = ?,
					status = 'active', is_demo = 1 WHERE user_id = ?""",
				(display_name, values[0], *values[2:], employee_code),
			)
		conn.execute("INSERT OR IGNORE INTO behavior_profiles(user_id) VALUES (?)", (employee_code,))
		conn.execute("INSERT OR IGNORE INTO restrictions(user_id) VALUES (?)", (employee_code,))
	for name, classification in SIMULATION_FILES:
		conn.execute("INSERT OR IGNORE INTO simulation_files(name, classification) VALUES (?, ?)", (name, classification))
	for device_id in ("USB-001", "USB-002", "Laptop-01"):
		conn.execute("INSERT OR IGNORE INTO simulation_devices(device_id, label) VALUES (?, ?)", (device_id, device_id))
	conn.commit()


def replace_analysis(frame: pd.DataFrame, db_path: str | Path = DEFAULT_DB) -> None:
	conn = connection(db_path)
	try:
		conn.execute("DELETE FROM alerts WHERE source = 'cert'")
		conn.execute("DELETE FROM results")
		feature_cols = [column for column in frame.columns if column not in {"user", "day", "anomaly_score", "is_flagged", "risk_score", "risk_level", "explanation", "alert_required", "anomaly_component"}]
		for row in frame.itertuples(index=False):
			values = row._asdict()
			conn.execute("INSERT INTO results(user_id, day, anomaly_score, is_flagged, risk_score, risk_level, explanation, features_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (values["user"], values["day"], float(values["anomaly_score"]), int(values["is_flagged"]), float(values["risk_score"]), values["risk_level"], values["explanation"], json.dumps({column: values[column] for column in feature_cols}, default=str)))
			if values["alert_required"]:
				conn.execute("INSERT INTO alerts(user_id, day, risk_score, risk_level, anomaly_score, explanation, source) VALUES (?, ?, ?, ?, ?, ?, 'cert')", (values["user"], values["day"], float(values["risk_score"]), values["risk_level"], float(values["anomaly_score"]), values["explanation"]))
		conn.commit()
	finally:
		conn.close()


def rows(query: str, params: tuple = (), db_path: str | Path = DEFAULT_DB) -> list[dict]:
	conn = connection(db_path)
	try:
		return [dict(row) for row in conn.execute(query, params).fetchall()]
	finally:
		conn.close()


def update_alert(alert_id: int, status: str, db_path: str | Path = DEFAULT_DB, source: str | None = None) -> bool:
	if status not in {"New", "Investigating", "Resolved"}:
		raise ValueError("Status must be New, Investigating, or Resolved")
	conn = connection(db_path)
	try:
		if source == "cert":
			cursor = conn.execute("UPDATE alerts SET status = ? WHERE id = ? AND source = 'cert'", (status, alert_id))
		elif source == "live":
			cursor = conn.execute("UPDATE alerts SET status = ? WHERE id = ? AND source <> 'cert'", (status, alert_id))
		else:
			cursor = conn.execute("UPDATE alerts SET status = ? WHERE id = ?", (status, alert_id))
		conn.commit()
		return cursor.rowcount == 1
	finally:
		conn.close()


def get_user(user_id: str, db_path: str | Path = DEFAULT_DB) -> dict | None:
	items = rows("SELECT user_id, display_name, role FROM users WHERE user_id = ?", (user_id,), db_path)
	return items[0] if items else None


def authenticate(user_id: str, password: str, db_path: str | Path = DEFAULT_DB) -> dict | None:
	conn = connection(db_path)
	try:
		row = conn.execute(
			"""SELECT user_id, display_name, role, password_hash FROM users
			WHERE user_id = ? OR employee_code = ? OR lower(email) = lower(?)""",
			(user_id, user_id, user_id),
		).fetchone()
		if not row or not verify_password(password, row["password_hash"]):
			return None
		return {"user_id": row["user_id"], "display_name": row["display_name"], "role": row["role"]}
	finally:
		conn.close()


def add_activity(user_id: str, event_type: str, timestamp: str, resource: str | None = None, metadata: dict | None = None, source: str = "simulation", db_path: str | Path = DEFAULT_DB) -> int:
	conn = connection(db_path)
	try:
		cursor = conn.execute("INSERT INTO activities(user_id, timestamp, event_type, resource, metadata_json, source) VALUES (?, ?, ?, ?, ?, ?)", (user_id, timestamp, event_type, resource, json.dumps(metadata or {}), source))
		conn.commit()
		return int(cursor.lastrowid)
	finally:
		conn.close()


def update_profile(user_id: str, event_type: str, event_hour: float, device_id: str | None = None, db_path: str | Path = DEFAULT_DB) -> dict:
	conn = connection(db_path)
	try:
		profile = conn.execute("SELECT * FROM behavior_profiles WHERE user_id = ?", (user_id,)).fetchone()
		known = set(json.loads(profile["known_devices_json"])) if profile else set()
		if device_id:
			known.add(device_id)
		count = profile["login_count"] if profile else 0
		old_hour = profile["typical_login_hour"] if profile and profile["typical_login_hour"] is not None else event_hour
		new_count = count + int(event_type == "LOGIN")
		new_hour = ((old_hour * count) + event_hour) / new_count if new_count else old_hour
		values = (new_hour, new_count, (profile["file_access_count"] if profile else 0) + int(event_type == "FILE_ACCESS"), (profile["download_count"] if profile else 0) + int(event_type == "FILE_DOWNLOAD"), (profile["device_count"] if profile else 0) + int(device_id is not None and device_id not in (json.loads(profile["known_devices_json"]) if profile else [])), (profile["offhours_count"] if profile else 0) + int(event_type == "LOGIN" and (event_hour < 6 or event_hour >= 21)), json.dumps(sorted(known)), user_id)
		conn.execute("UPDATE behavior_profiles SET typical_login_hour = ?, login_count = ?, file_access_count = ?, download_count = ?, device_count = ?, offhours_count = ?, known_devices_json = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?", values)
		conn.commit()
		return dict(conn.execute("SELECT * FROM behavior_profiles WHERE user_id = ?", (user_id,)).fetchone())
	finally:
		conn.close()


def set_policy(user_id: str, *, file_access: int | None = None, confidential_access: int | None = None, downloads: int | None = None, usb_access: int | None = None, monitoring_state: str | None = None, db_path: str | Path = DEFAULT_DB) -> None:
	conn = connection(db_path)
	try:
		if file_access is not None:
			conn.execute("UPDATE restrictions SET file_access = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?", (file_access, user_id))
		for column, value in (("confidential_access", confidential_access), ("downloads", downloads), ("usb_access", usb_access)):
			if value is not None:
				conn.execute(f"UPDATE restrictions SET {column} = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?", (value, user_id))
		if monitoring_state is not None:
			conn.execute("UPDATE restrictions SET monitoring_state = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?", (monitoring_state, user_id))
		conn.commit()
	finally:
		conn.close()
