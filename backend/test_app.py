import json
import unittest

import pandas as pd

from backend.analytics import EVALUATION_PATH, prepare_model_data, summarize_matches
from backend.app import analytics_data, app, model


def sample_matches():
    return pd.DataFrame(
        [
            {
                "id": 1,
                "season": "2007/08",
                "date": "2008-04-18",
                "venue": "M Chinnaswamy Stadium",
                "team1": "Chennai Super Kings",
                "team2": "Mumbai Indians",
                "toss_winner": "Chennai Super Kings",
                "toss_decision": "bat",
                "winner": "Chennai Super Kings",
                "result": "runs",
            },
            {
                "id": 2,
                "season": "2007/08",
                "date": "2008-04-19",
                "venue": "Eden Gardens",
                "team1": "Mumbai Indians",
                "team2": "Chennai Super Kings",
                "toss_winner": "Mumbai Indians",
                "toss_decision": "field",
                "winner": "Chennai Super Kings",
                "result": "wickets",
            },
            {
                "id": 3,
                "season": "2007/08",
                "date": "2008-04-20",
                "venue": "Eden Gardens",
                "team1": "Chennai Super Kings",
                "team2": "Mumbai Indians",
                "toss_winner": "Mumbai Indians",
                "toss_decision": "bat",
                "winner": None,
                "result": "no result",
            },
        ]
    )


class DatasetAnalyticsTests(unittest.TestCase):
    def test_summary_calculates_match_counts_and_win_percentage(self):
        summary = summarize_matches(sample_matches(), delivery_count=10, delivery_orphan_count=0)

        self.assertEqual(summary["matches"], 3)
        self.assertEqual(summary["completed_matches"], 2)
        self.assertEqual(summary["no_result_matches"], 1)
        self.assertEqual(summary["deliveries"], 10)
        self.assertEqual(summary["toss"]["winner_wins"], 1)
        self.assertEqual(summary["toss"]["winner_loses"], 1)
        self.assertEqual(summary["toss"]["win_percentage"], 50.0)
        self.assertEqual(summary["team_rankings"][0]["team"], "Chennai Super Kings")
        self.assertEqual(summary["team_rankings"][0]["matches_played"], 3)
        self.assertEqual(summary["team_rankings"][0]["wins"], 2)
        self.assertEqual(summary["team_rankings"][0]["win_percentage"], 66.67)

    def test_model_preparation_excludes_no_result_and_derives_toss_features(self):
        prepared = prepare_model_data(sample_matches())

        self.assertEqual(len(prepared), 2)
        self.assertEqual(prepared["toss_winner_is_team1"].tolist(), [1, 1])
        self.assertEqual(prepared["team1_batting_first"].tolist(), [1, 0])


class FlaskIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    @unittest.skipIf(analytics_data is None, "matches.csv is not configured")
    def test_analytics_endpoint_returns_data_derived_counts(self):
        response = self.client.get("/api/analytics")

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["matches"], 1095)
        self.assertEqual(data["deliveries"], 260920)
        self.assertEqual(data["completed_matches"], 1090)
        self.assertEqual(data["data_quality"]["duplicate_match_ids"], 0)
        self.assertEqual(data["data_quality"]["delivery_orphan_match_ids"], 0)
        self.assertTrue(data["team_rankings"])
        self.assertTrue(data["top_venues"])

    def test_health_reports_model_and_dataset_state(self):
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["model_loaded"], model is not None)

    def test_metrics_use_generated_evaluation_report(self):
        response = self.client.get("/api/metrics")

        self.assertEqual(response.status_code, 200)
        metrics = response.get_json()
        if metrics["evaluation_available"]:
            report = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
            self.assertEqual(metrics["random_forest_accuracy"], report["random_forest_accuracy"])
            self.assertEqual(
                metrics["toss_winner_baseline_accuracy"],
                report["toss_winner_baseline_accuracy"],
            )
            self.assertEqual(metrics["test_matches"], report["test_matches"])
        else:
            self.assertIsNone(metrics["random_forest_accuracy"])

    @unittest.skipIf(model is None, "trained model artifact is unavailable")
    def test_prediction_uses_supported_categories(self):
        options = self.client.get("/api/options").get_json()
        response = self.client.post(
            "/api/predict",
            json={
                "season": options["seasons_seen_in_training"][-1],
                "team1": options["teams_supported"][0],
                "team2": options["teams_supported"][1],
                "venue": options["venues_seen_in_training"][0],
                "toss_winner_is_team1": "1",
                "toss_decision": options["toss_decisions"][0],
                "team1_batting_first": "1",
            },
        )

        self.assertEqual(response.status_code, 200)
        prediction = response.get_json()
        self.assertIn(prediction["predicted_winner"], options["teams_supported"])
        self.assertIn("chronological test matches", prediction["notice"])
        self.assertIn("not a calibrated probability", prediction["notice"])

    @unittest.skipIf(model is None, "trained model artifact is unavailable")
    def test_prediction_rejects_unknown_seasons(self):
        options = self.client.get("/api/options").get_json()
        response = self.client.post(
            "/api/predict",
            json={
                "season": "2099",
                "team1": options["teams_supported"][0],
                "team2": options["teams_supported"][1],
                "venue": options["venues_seen_in_training"][0],
                "toss_winner_is_team1": 1,
                "toss_decision": options["toss_decisions"][0],
                "team1_batting_first": 1,
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported season", response.get_json()["message"])

    def test_report_assets_are_served(self):
        for path, content_type in [
            ("/reports/previews/page1.png", "image/png"),
            ("/reports/previews/page2.png", "image/png"),
            ("/reports/ipl-dashboard.pbix", "application/octet-stream"),
        ]:
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.content_type, content_type)
                self.assertTrue(response.data)
                response.close()


if __name__ == "__main__":
    unittest.main()
