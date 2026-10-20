"""Unit tests for SentinelAI SQLite database functionality and schema indexes."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from backend import database


class TestDatabaseSecurityAndSchema(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_sentinelai.db"
        self.conn = database.connection(self.db_path)

    def tearDown(self):
        self.conn.close()
        self.temp_dir.cleanup()

    def test_password_hashing_and_verification(self):
        password = "SecurePassword123!"
        hashed = database.hash_password(password)

        self.assertIn("$", hashed)
        self.assertTrue(database.verify_password(password, hashed))
        self.assertFalse(database.verify_password("WrongPassword", hashed))
        self.assertFalse(database.verify_password("", hashed))
        self.assertFalse(database.verify_password(password, "invalid_hash_string"))

    def test_database_initialization_creates_tables(self):
        tables = {
            row["name"]
            for row in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        expected_tables = {
            "users",
            "activities",
            "behavior_profiles",
            "risk_history",
            "restrictions",
            "simulation_files",
            "simulation_devices",
            "results",
            "alerts",
        }
        self.assertTrue(expected_tables.issubset(tables))

    def test_performance_indexes_exist(self):
        indexes = {
            row["name"]
            for row in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        expected_indexes = {
            "idx_activities_user_timestamp",
            "idx_activities_user_resource",
            "idx_risk_history_user_timestamp",
            "idx_risk_history_event_id",
            "idx_alerts_source_risk",
            "idx_results_user_day",
        }
        for index_name in expected_indexes:
            self.assertIn(index_name, indexes, f"Index {index_name} was not created")

    def test_admin_user_seeded(self):
        admin = database.authenticate("admin", "admin-demo", self.db_path)
        self.assertIsNotNone(admin)
        self.assertEqual(admin["user_id"], "admin")
        self.assertEqual(admin["role"], "admin")

    def test_add_activity_and_query_rows(self):
        activity_id = database.add_activity(
            user_id="EMP-003",
            event_type="FILE_ACCESS",
            timestamp="2026-10-20T10:00:00Z",
            resource="project_report.pdf",
            metadata={"test": True},
            source="simulation",
            db_path=self.db_path,
        )
        self.assertIsInstance(activity_id, int)
        self.assertGreater(activity_id, 0)

        records = database.rows(
            "SELECT * FROM activities WHERE user_id = ?",
            ("EMP-003",),
            db_path=self.db_path,
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["event_type"], "FILE_ACCESS")
        self.assertEqual(records[0]["resource"], "project_report.pdf")

    def test_update_profile_handles_existing_and_new_users(self):
        profile = database.update_profile(
            user_id="EMP-003",
            event_type="LOGIN",
            event_hour=9.0,
            db_path=self.db_path,
        )
        self.assertIsNotNone(profile)
        self.assertGreaterEqual(profile["login_count"], 1)

        new_user_profile = database.update_profile(
            user_id="NEW_EMP_999",
            event_type="LOGIN",
            event_hour=14.0,
            device_id="USB-009",
            db_path=self.db_path,
        )
        self.assertIsNotNone(new_user_profile)
        self.assertEqual(new_user_profile["login_count"], 1)

    def test_update_alert_status(self):
        self.conn.execute(
            """INSERT INTO alerts(user_id, day, risk_score, risk_level, anomaly_score, explanation, source)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("EMP-003", "2026-10-20", 80.0, "Critical", 0.9, "High risk activity", "cert"),
        )
        self.conn.commit()

        alert_id = self.conn.execute("SELECT id FROM alerts WHERE user_id = 'EMP-003'").fetchone()["id"]

        success = database.update_alert(alert_id, "Investigating", db_path=self.db_path, source="cert")
        self.assertTrue(success)

        with self.assertRaises(ValueError):
            database.update_alert(alert_id, "InvalidStatus", db_path=self.db_path)


if __name__ == "__main__":
    unittest.main()
