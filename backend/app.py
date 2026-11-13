from pathlib import Path
import json
import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

if __package__:
    from backend.analytics import EVALUATION_PATH, load_dataset_snapshot, sha256_file
else:
    from analytics import EVALUATION_PATH, load_dataset_snapshot, sha256_file

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
MODEL_PATH = ROOT / "models" / "ipl_winner_model.pkl"

# The supplied artifact was saved with scikit-learn 1.6.1.
# Keep this compatibility alias for newer sklearn versions where the
# internal helper class was renamed/removed. requirements.txt pins 1.6.1.
try:
    import sklearn.compose._column_transformer as _ct
    if not hasattr(_ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        _ct._RemainderColsList = _RemainderColsList
except Exception:
    pass

app = Flask(__name__, static_folder=str(FRONTEND), static_url_path="")
model = None
model_load_error = None

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    model_load_error = str(exc)

matches_data = None
analytics_data = None
dataset_load_error = None
dataset_paths = {}
dataset_hashes = {}
try:
    matches_data, analytics_data, dataset_paths = load_dataset_snapshot()
    dataset_hashes = {
        key: sha256_file(path)
        for key, path in dataset_paths.items()
        if path is not None
    }
except Exception as exc:
    dataset_load_error = str(exc)

evaluation_data = None
evaluation_load_error = None
try:
    if EVALUATION_PATH.is_file():
        evaluation_data = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    evaluation_load_error = str(exc)

evaluation_matches_data = bool(evaluation_data) and (
    "matches" not in dataset_hashes
    or evaluation_data.get("matches_sha256") == dataset_hashes["matches"]
)
model_current = model is not None and (
    "matches" not in dataset_hashes or evaluation_matches_data
)
if model is not None and "matches" in dataset_hashes and not evaluation_matches_data:
    model_load_error = "The loaded model evaluation does not match the currently configured matches.csv. Rerun backend\\analytics.py."

@app.get("/")
def home():
    return send_from_directory(FRONTEND, "index.html")

@app.get("/reports/previews/<path:filename>")
def report_preview(filename):
    if filename not in {"page1.png", "page2.png"}:
        return jsonify({"message": "Preview not found."}), 404
    return send_from_directory(ROOT / "Dashboard_Screenshots", filename)

@app.get("/reports/ipl-dashboard.pbix")
def power_bi_report():
    return send_from_directory(
        ROOT / "PowerBI",
        "IPL_DASHBOARD_REPORT.pbix",
        as_attachment=True,
        download_name="IPL_DASHBOARD_REPORT.pbix"
    )

@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok" if model_current and analytics_data is not None else "degraded",
        "model_loaded": model is not None,
        "model_current": model_current,
        "model_error": model_load_error,
        "dataset_loaded": analytics_data is not None,
        "dataset_error": dataset_load_error
    })

@app.get("/api/metrics")
def metrics():
    evaluation = evaluation_data if evaluation_matches_data else {}
    rolling = evaluation.get("rolling_origin", {})
    return jsonify({
        "matches": analytics_data["matches"] if analytics_data else None,
        "deliveries": analytics_data["deliveries"] if analytics_data else None,
        "dataset_period": analytics_data["dataset_period"] if analytics_data else None,
        "random_forest_accuracy": evaluation.get("random_forest_accuracy"),
        "random_forest_accuracy_95_ci": evaluation.get("random_forest_accuracy_95_ci"),
        "toss_winner_baseline_accuracy": evaluation.get("toss_winner_baseline_accuracy"),
        "toss_winner_baseline_95_ci": evaluation.get("toss_winner_baseline_95_ci"),
        "test_matches": evaluation.get("test_matches"),
        "rolling_origin_accuracy": rolling.get("accuracy"),
        "rolling_origin_accuracy_95_ci": rolling.get("accuracy_95_ci"),
        "rolling_origin_toss_baseline_accuracy": rolling.get("toss_winner_baseline_accuracy"),
        "rolling_origin_matches": rolling.get("match_count"),
        "evaluation_available": bool(evaluation),
        "evaluation_error": (
            evaluation_load_error
            or (
                "The evaluation report does not match the configured matches.csv. Rerun backend\\analytics.py."
                if evaluation_data and not evaluation_matches_data
                else None
            )
        ),
        "evaluation_protocol": evaluation.get("evaluation_protocol"),
        "matches_with_unseen_team": evaluation.get("matches_with_unseen_team"),
        "matches_with_unseen_winner": evaluation.get("matches_with_unseen_winner"),
        "model_training_end_date": evaluation.get("model_training_end_date"),
        "limitations": evaluation.get("limitations", []),
        "data_provenance": evaluation.get("data_provenance")
    })

@app.get("/api/analytics")
def analytics():
    if analytics_data is None:
        return jsonify({
            "error": "Historical analytics are unavailable.",
            "detail": dataset_load_error
        }), 503
    return jsonify(analytics_data)

@app.get("/api/options")
def options():
    if not model_current:
        return jsonify({"error": "Model is not loaded or is out of date", "detail": model_load_error}), 503
    encoder = model.named_steps["preprocessor"].named_transformers_["categorical"]
    cats = encoder.categories_
    return jsonify({
        "seasons_seen_in_training": [str(x) for x in cats[0]],
        "teams_supported": [str(x) for x in model.classes_],
        "venues_seen_in_training": [str(x) for x in cats[3]],
        "toss_decisions": [str(x) for x in cats[4]]
    })

@app.post("/api/predict")
def predict():
    if not model_current:
        return jsonify({"status": "model_unavailable", "message": "The model could not be loaded. Check the pinned scikit-learn version.", "detail": model_load_error}), 503

    data = request.get_json(silent=True) or {}
    required = ["season", "team1", "team2", "venue", "toss_winner_is_team1", "toss_decision", "team1_batting_first"]
    missing = [key for key in required if key not in data or data[key] in (None, "")]
    if missing:
        return jsonify({"message": "Missing fields: " + ", ".join(missing)}), 400
    if data["team1"] == data["team2"]:
        return jsonify({"message": "Select two different teams."}), 400

    encoder = model.named_steps["preprocessor"].named_transformers_["categorical"]
    categories = encoder.categories_
    supported_values = {
        "season": {str(value) for value in categories[0]},
        "team1": {str(value) for value in model.classes_},
        "team2": {str(value) for value in model.classes_},
        "venue": {str(value) for value in categories[3]},
        "toss_decision": {str(value) for value in categories[4]}
    }
    for field, values in supported_values.items():
        if str(data[field]) not in values:
            return jsonify({"message": f"Unsupported {field.replace('_', ' ')}. Choose a value returned by /api/options."}), 400

    try:
        toss_winner_is_team1 = int(data["toss_winner_is_team1"])
        team1_batting_first = int(data["team1_batting_first"])
    except (TypeError, ValueError):
        return jsonify({"message": "Toss winner and batting-first selections must be Team 1 or Team 2."}), 400
    if toss_winner_is_team1 not in (0, 1) or team1_batting_first not in (0, 1):
        return jsonify({"message": "Toss winner and batting-first selections must be Team 1 or Team 2."}), 400

    try:
        row = pd.DataFrame([{
            "season": str(data["season"]),
            "team1": str(data["team1"]),
            "team2": str(data["team2"]),
            "venue": str(data["venue"]),
            "toss_winner_is_team1": toss_winner_is_team1,
            "toss_decision": str(data["toss_decision"]),
            "team1_batting_first": team1_batting_first
        }], columns=required)
        predicted = str(model.predict(row)[0])
        probabilities = model.predict_proba(row)[0]
        score = float(max(probabilities))
        if evaluation_data and evaluation_matches_data:
            accuracy = evaluation_data["random_forest_accuracy"] * 100
            baseline = evaluation_data["toss_winner_baseline_accuracy"] * 100
            test_matches = evaluation_data["test_matches"]
            notice = (
                f"Experimental historical model. It scored {accuracy:.2f}% on "
                f"{test_matches} chronological test matches, below the "
                f"{baseline:.2f}% toss-winner baseline. This score is not a calibrated probability."
            )
        else:
            notice = (
                "Experimental historical model. Evaluation metrics are unavailable; "
                "this score is not a calibrated probability."
            )
        return jsonify({
            "status": "success",
            "predicted_winner": predicted,
            "model_score": round(score * 100, 1),
            "notice": notice
        })
    except Exception as exc:
        return jsonify({"status": "error", "message": "Prediction failed for these inputs.", "detail": str(exc)}), 400

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
