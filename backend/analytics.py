import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "Dataset"
MODEL_PATH = ROOT / "models" / "ipl_winner_model.pkl"
EVALUATION_PATH = ROOT / "models" / "model_evaluation.json"

TEAM_NAME_MAPPING = {
    "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}

FEATURE_COLUMNS = [
    "season",
    "team1",
    "team2",
    "venue",
    "toss_winner_is_team1",
    "toss_decision",
    "team1_batting_first",
]
CATEGORICAL_FEATURES = ["season", "team1", "team2", "venue", "toss_decision"]
NUMERICAL_FEATURES = ["toss_winner_is_team1", "team1_batting_first"]
REQUIRED_MATCH_COLUMNS = {
    "id",
    "season",
    "date",
    "venue",
    "team1",
    "team2",
    "toss_winner",
    "toss_decision",
    "winner",
    "result",
}
REQUIRED_DELIVERY_COLUMNS = {"match_id"}


def resolve_data_file(filename, environment_variable, required=True):
    configured = os.environ.get(environment_variable)
    if configured:
        path = Path(configured).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{environment_variable} does not point to a file: {path}")
        return path.resolve()

    candidates = [DATASET_DIR / filename, Path.home() / "Downloads" / filename]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    if required:
        raise FileNotFoundError(
            f"{filename} was not found. Set {environment_variable} or place it in Dataset/."
        )
    return None


def canonicalize_matches(matches):
    missing = REQUIRED_MATCH_COLUMNS.difference(matches.columns)
    if missing:
        raise ValueError(f"matches.csv is missing required columns: {', '.join(sorted(missing))}")

    clean = matches.copy()
    for column in ["team1", "team2", "toss_winner", "winner"]:
        clean[column] = clean[column].replace(TEAM_NAME_MAPPING)
    clean["date"] = pd.to_datetime(clean["date"], errors="coerce")
    clean["season"] = clean["season"].astype("string")
    return clean


def prepare_model_data(matches):
    clean = canonicalize_matches(matches)
    required_features = [
        "date",
        "season",
        "venue",
        "team1",
        "team2",
        "toss_winner",
        "toss_decision",
        "winner",
    ]
    model_data = clean.dropna(subset=required_features).copy()
    model_data = model_data.sort_values("date", kind="stable").reset_index(drop=True)
    model_data["toss_winner_is_team1"] = (
        model_data["toss_winner"] == model_data["team1"]
    ).astype(int)
    model_data["team1_batting_first"] = (
        (
            (model_data["toss_winner"] == model_data["team1"])
            & (model_data["toss_decision"] == "bat")
        )
        | (
            (model_data["toss_winner"] == model_data["team2"])
            & (model_data["toss_decision"] == "field")
        )
    ).astype(int)
    return model_data


def create_model():
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numerical", "passthrough", NUMERICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def wilson_interval(successes, total, z=1.96):
    if total == 0:
        return None
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    radius = (
        z
        * np.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return [round(max(0.0, center - radius), 4), round(min(1.0, center + radius), 4)]


def evaluate_predictions(model, train, test):
    actual = test["winner"]
    predicted = model.predict(test[FEATURE_COLUMNS])
    toss_predictions = np.where(
        test["toss_winner_is_team1"].to_numpy() == 1,
        test["team1"].to_numpy(),
        test["team2"].to_numpy(),
    )
    majority_team = train["winner"].value_counts().index[0]
    known_training_teams = set(train["team1"]) | set(train["team2"])
    unseen_matchups = ~test["team1"].isin(known_training_teams) | ~test[
        "team2"
    ].isin(known_training_teams)
    unseen_winners = ~actual.isin(train["winner"].unique())
    accuracy = accuracy_score(actual, predicted)
    toss_accuracy = accuracy_score(actual, toss_predictions)
    report_labels = sorted(set(actual.unique()) | set(predicted))
    correct = int((actual.to_numpy() == predicted).sum())
    toss_correct = int((actual.to_numpy() == toss_predictions).sum())
    return {
        "match_count": len(test),
        "start_date": test["date"].min().date().isoformat(),
        "end_date": test["date"].max().date().isoformat(),
        "accuracy": round(float(accuracy), 4),
        "correct_predictions": correct,
        "accuracy_95_ci": wilson_interval(correct, len(test)),
        "macro_f1": round(
            float(f1_score(actual, predicted, average="macro", zero_division=0)), 4
        ),
        "weighted_f1": round(
            float(f1_score(actual, predicted, average="weighted", zero_division=0)), 4
        ),
        "toss_winner_baseline_accuracy": round(float(toss_accuracy), 4),
        "toss_winner_baseline_correct": toss_correct,
        "toss_winner_baseline_95_ci": wilson_interval(toss_correct, len(test)),
        "majority_team_baseline_accuracy": round(
            float(accuracy_score(actual, np.repeat(majority_team, len(test)))), 4
        ),
        "model_minus_toss_baseline_percentage_points": round(
            float((accuracy - toss_accuracy) * 100), 2
        ),
        "matches_with_unseen_team": int(unseen_matchups.sum()),
        "matches_with_unseen_winner": int(unseen_winners.sum()),
        "classification_report": classification_report(
            actual,
            predicted,
            labels=report_labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def rolling_origin_evaluation(model_data, season_count=5):
    seasons = model_data.drop_duplicates("season")["season"].tolist()
    seasons = seasons[-season_count:]
    folds = []
    actual_all = []
    predicted_all = []
    toss_all = []

    for season in seasons:
        test = model_data.loc[model_data["season"] == season]
        first_test_date = test["date"].min()
        train = model_data.loc[model_data["date"] < first_test_date]
        if train.empty or test.empty:
            continue
        model = create_model()
        model.fit(train[FEATURE_COLUMNS], train["winner"])
        fold = evaluate_predictions(model, train, test)
        fold["season"] = str(season)
        folds.append(fold)
        actual_all.extend(test["winner"].tolist())
        predicted_all.extend(model.predict(test[FEATURE_COLUMNS]).tolist())
        toss_all.extend(
            np.where(
                test["toss_winner_is_team1"].to_numpy() == 1,
                test["team1"].to_numpy(),
                test["team2"].to_numpy(),
            ).tolist()
        )

    total = len(actual_all)
    if not total:
        return {"folds": folds, "match_count": 0}
    correct = sum(actual == predicted for actual, predicted in zip(actual_all, predicted_all))
    toss_correct = sum(actual == predicted for actual, predicted in zip(actual_all, toss_all))
    return {
        "folds": folds,
        "match_count": total,
        "accuracy": round(correct / total, 4),
        "correct_predictions": correct,
        "accuracy_95_ci": wilson_interval(correct, total),
        "toss_winner_baseline_accuracy": round(toss_correct / total, 4),
        "toss_winner_baseline_correct": toss_correct,
        "toss_winner_baseline_95_ci": wilson_interval(toss_correct, total),
        "model_minus_toss_baseline_percentage_points": round(
            (correct - toss_correct) / total * 100, 2
        ),
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_evaluate(matches, model_path=MODEL_PATH, evaluation_path=EVALUATION_PATH, data_files=None):
    model_data = prepare_model_data(matches)
    if len(model_data) < 10:
        raise ValueError("At least 10 matches with winners are required to evaluate the model.")

    split_index = int(len(model_data) * 0.8)
    train = model_data.iloc[:split_index]
    test = model_data.iloc[split_index:]
    if train.empty or test.empty:
        raise ValueError("The chronological 80/20 split produced an empty partition.")

    evaluation_model = create_model()
    evaluation_model.fit(train[FEATURE_COLUMNS], train["winner"])
    holdout = evaluate_predictions(evaluation_model, train, test)
    evaluation = {
        "evaluation_protocol": "Chronological 80/20 holdout; rows with no winner are excluded.",
        "feature_columns": FEATURE_COLUMNS,
        "training_matches": len(train),
        "test_matches": len(test),
        "training_start_date": train["date"].min().date().isoformat(),
        "training_end_date": train["date"].max().date().isoformat(),
        "test_start_date": test["date"].min().date().isoformat(),
        "test_end_date": test["date"].max().date().isoformat(),
        "random_forest_accuracy": holdout["accuracy"],
        "random_forest_correct": holdout["correct_predictions"],
        "random_forest_accuracy_95_ci": holdout["accuracy_95_ci"],
        "toss_winner_baseline_accuracy": holdout["toss_winner_baseline_accuracy"],
        "toss_winner_baseline_correct": holdout["toss_winner_baseline_correct"],
        "toss_winner_baseline_95_ci": holdout["toss_winner_baseline_95_ci"],
        "majority_team_baseline_accuracy": holdout["majority_team_baseline_accuracy"],
        "model_minus_toss_baseline_percentage_points": holdout[
            "model_minus_toss_baseline_percentage_points"
        ],
        "macro_f1": holdout["macro_f1"],
        "weighted_f1": holdout["weighted_f1"],
        "matches_with_unseen_team": holdout["matches_with_unseen_team"],
        "matches_with_unseen_winner": holdout["matches_with_unseen_winner"],
        "classification_report": holdout["classification_report"],
        "rolling_origin": rolling_origin_evaluation(model_data),
        "limitations": [
            "The model has not demonstrated better accuracy than the toss-winner baseline.",
            "The chronological test set is small and spans only the latest seasons in this dataset.",
            "Teams and seasons absent from training cannot be reliably predicted.",
            "The winning-team class score is not a calibrated probability.",
            "The exact Kaggle dataset and its license have not been identified; verify usage rights before redistribution.",
        ],
        "data_provenance": "Kaggle (exact dataset and URL not provided); license unknown.",
        "model_training_rows": len(model_data),
        "model_training_start_date": model_data["date"].min().date().isoformat(),
        "model_training_end_date": model_data["date"].max().date().isoformat(),
    }

    for key, path in (data_files or {}).items():
        if path is not None:
            evaluation[f"{key}_file"] = Path(path).name
            evaluation[f"{key}_sha256"] = sha256_file(Path(path))

    production_model = create_model()
    production_model.fit(model_data[FEATURE_COLUMNS], model_data["winner"])
    model_path = Path(model_path)
    evaluation_path = Path(evaluation_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(production_model, model_path)
    evaluation_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    return evaluation


def summarize_matches(matches, delivery_count=None, delivery_orphan_count=None):
    clean = canonicalize_matches(matches)
    team_counts = pd.concat([clean["team1"], clean["team2"]]).value_counts()
    wins = clean["winner"].value_counts()
    teams = []
    for team, played in team_counts.items():
        team_wins = int(wins.get(team, 0))
        teams.append(
            {
                "team": str(team),
                "matches_played": int(played),
                "wins": team_wins,
                "win_percentage": round(team_wins / int(played) * 100, 2),
            }
        )
    teams.sort(key=lambda row: (-row["wins"], -row["win_percentage"], row["team"]))

    completed = clean.dropna(subset=["winner"])
    toss_won = int((completed["toss_winner"] == completed["winner"]).sum())
    toss_decisions = {
        str(key): int(value)
        for key, value in clean["toss_decision"].value_counts().items()
    }
    results = {
        str(key): int(value)
        for key, value in clean["result"].value_counts(dropna=False).items()
    }
    season_matches = (
        clean.groupby("season", dropna=False)
        .size()
        .rename_axis("season")
        .reset_index(name="matches")
        .to_dict("records")
    )
    season_matches = [
        {"season": str(row["season"]), "matches": int(row["matches"])}
        for row in season_matches
    ]
    season_matches.sort(key=lambda row: row["season"])
    venues = [
        {"venue": str(venue), "matches": int(count)}
        for venue, count in clean["venue"].value_counts().head(10).items()
    ]
    dates = clean["date"].dropna()

    return {
        "dataset_period": (
            f"{dates.min().year}-{dates.max().year}" if not dates.empty else None
        ),
        "matches": len(clean),
        "deliveries": delivery_count,
        "completed_matches": len(completed),
        "no_result_matches": int(clean["winner"].isna().sum()),
        "team_rankings": teams,
        "top_venues": venues,
        "toss": {
            "winner_wins": toss_won,
            "winner_loses": len(completed) - toss_won,
            "no_result": int(clean["winner"].isna().sum()),
            "win_percentage": round(toss_won / len(completed) * 100, 2)
            if len(completed)
            else None,
            "decisions": toss_decisions,
        },
        "results": results,
        "season_matches": season_matches,
        "data_quality": {
            "duplicate_match_ids": int(clean["id"].duplicated().sum()),
            "delivery_orphan_match_ids": delivery_orphan_count,
            "missing_winner_matches": int(clean["winner"].isna().sum()),
            "provenance": "Kaggle (exact dataset and URL not provided); license unknown.",
        },
    }


def load_dataset_snapshot():
    matches_path = resolve_data_file("matches.csv", "IPL_MATCHES_CSV")
    deliveries_path = resolve_data_file(
        "deliveries.csv", "IPL_DELIVERIES_CSV", required=False
    )
    matches = pd.read_csv(matches_path)
    summary = summarize_matches(matches)
    if deliveries_path is not None:
        delivery_columns = set(pd.read_csv(deliveries_path, nrows=0).columns)
        missing_columns = REQUIRED_DELIVERY_COLUMNS.difference(delivery_columns)
        if missing_columns:
            raise ValueError(f"deliveries.csv is missing required columns: {', '.join(sorted(missing_columns))}")
        delivery_ids = pd.read_csv(deliveries_path, usecols=["match_id"])["match_id"]
        known_ids = set(matches["id"].dropna())
        orphan_count = int((~delivery_ids.isin(known_ids)).sum())
        summary["deliveries"] = len(delivery_ids)
        summary["data_quality"]["delivery_orphan_match_ids"] = orphan_count
    return matches, summary, {"matches": matches_path, "deliveries": deliveries_path}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the IPL winner model chronologically and train a full-data pipeline."
    )
    parser.add_argument("--matches", type=Path, help="Path to the supplied matches.csv")
    parser.add_argument("--deliveries", type=Path, help="Path to the supplied deliveries.csv")
    parser.add_argument("--model-output", type=Path, default=MODEL_PATH)
    parser.add_argument("--evaluation-output", type=Path, default=EVALUATION_PATH)
    args = parser.parse_args()

    matches_path = args.matches or resolve_data_file("matches.csv", "IPL_MATCHES_CSV")
    deliveries_path = args.deliveries or resolve_data_file(
        "deliveries.csv", "IPL_DELIVERIES_CSV", required=False
    )
    matches = pd.read_csv(matches_path)
    files = {"matches": matches_path, "deliveries": deliveries_path}
    evaluation = train_evaluate(
        matches,
        model_path=args.model_output,
        evaluation_path=args.evaluation_output,
        data_files=files,
    )
    print(
        f"Chronological holdout: {evaluation['random_forest_correct']}/"
        f"{evaluation['test_matches']} = {evaluation['random_forest_accuracy']:.2%}"
    )
    print(
        "Toss-winner baseline: "
        f"{evaluation['toss_winner_baseline_correct']}/{evaluation['test_matches']} = "
        f"{evaluation['toss_winner_baseline_accuracy']:.2%}"
    )
    rolling = evaluation["rolling_origin"]
    print(
        f"Rolling-origin: {rolling.get('correct_predictions', 0)}/"
        f"{rolling.get('match_count', 0)} = {rolling.get('accuracy', 0):.2%}; "
        "toss baseline = "
        f"{rolling.get('toss_winner_baseline_accuracy', 0):.2%}"
    )
    print(f"Saved trained model: {args.model_output}")
    print(f"Saved evaluation: {args.evaluation_output}")


if __name__ == "__main__":
    main()
