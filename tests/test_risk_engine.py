"""Unit tests for SentinelAI heuristic risk scoring engine."""

import unittest

from backend.risk_engine import _explain, _level, simulation_risk


class TestRiskEngine(unittest.TestCase):
    def test_level_mapping(self):
        self.assertEqual(_level(100.0), "Critical")
        self.assertEqual(_level(75.0), "Critical")
        self.assertEqual(_level(74.9), "High")
        self.assertEqual(_level(50.0), "High")
        self.assertEqual(_level(49.9), "Medium")
        self.assertEqual(_level(25.0), "Medium")
        self.assertEqual(_level(24.9), "Low")
        self.assertEqual(_level(0.0), "Low")

    def test_simulation_risk_normal_activity(self):
        profile = {"login_count": 5}
        score, level, explanation = simulation_risk(
            profile=profile,
            event_type="LOGIN",
            event_hour=10,
            resource=None,
            known_resource=True,
            known_device=True,
            recent_suspicious=0,
        )
        self.assertEqual(score, 0.0)
        self.assertEqual(level, "Low")
        self.assertIn("consistent with the observed simulation baseline", explanation)

    def test_simulation_risk_offhours_login(self):
        profile = {"login_count": 5}
        score, level, explanation = simulation_risk(
            profile=profile,
            event_type="LOGIN",
            event_hour=23,
            resource=None,
            known_resource=True,
            known_device=True,
            recent_suspicious=0,
        )
        self.assertGreaterEqual(score, 22.0)
        self.assertIn("outside the user's normal activity window", explanation)

    def test_simulation_risk_unusual_resource_and_new_device(self):
        profile = {"login_count": 5}
        score, level, explanation = simulation_risk(
            profile=profile,
            event_type="FILE_ACCESS",
            event_hour=14,
            resource="confidential_strategy.pdf",
            known_resource=False,
            known_device=False,
            recent_suspicious=0,
        )
        self.assertEqual(score, 18.0)
        self.assertIn("resource is unusual", explanation)

        score_dev, level_dev, explanation_dev = simulation_risk(
            profile=profile,
            event_type="DEVICE_CONNECT",
            event_hour=14,
            resource="USB-999",
            known_resource=False,
            known_device=False,
            recent_suspicious=3,
        )
        # 20 (new device) + 15 (3 * 5 recent) = 35
        self.assertEqual(score_dev, 35.0)
        self.assertEqual(level_dev, "Medium")
        self.assertIn("device not previously observed", explanation_dev)
        self.assertIn("multiple unusual events occurred recently", explanation_dev)

    def test_explain_fallback_and_indicators(self):
        empty_row = {}
        self.assertEqual(
            _explain(empty_row),
            "No strong contextual indicators were observed in this user-day.",
        )

        row_with_indicators = {
            "risk_level": "High",
            "anomaly_component": 0.8,
            "offhours_logon": 1,
            "n_distinct_pcs": 3,
            "file_events": 12,
        }
        explanation = _explain(row_with_indicators)
        self.assertTrue(explanation.startswith("High risk because "))
        self.assertIn("the activity was anomalous", explanation)
        self.assertIn("outside typical hours", explanation)
        self.assertIn("involved 3 computers", explanation)
        self.assertIn("12 file events were recorded", explanation)


if __name__ == "__main__":
    unittest.main()
