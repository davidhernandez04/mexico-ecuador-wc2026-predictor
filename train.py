"""Train outcome and score models on international match history."""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLUMNS, build_training_frame, load_results

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "results.csv"
MODEL_DIR = ROOT / "models"


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_results(str(DATA_PATH))
    train_df = build_training_frame(raw, min_year=1990)

    extra_cols = []
    feature_cols = FEATURE_COLUMNS + extra_cols
    X = train_df[feature_cols]
    y_outcome = train_df["outcome"]
    y_home_goals = train_df["home_score"]
    y_away_goals = train_df["away_score"]

    X_train, X_test, y_out_train, y_out_test, y_h_train, y_h_test, y_a_train, y_a_test = train_test_split(
        X,
        y_outcome,
        y_home_goals,
        y_away_goals,
        test_size=0.15,
        random_state=42,
        stratify=y_outcome,
    )

    outcome_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=250,
                    learning_rate=0.05,
                    max_depth=4,
                    random_state=42,
                ),
            ),
        ]
    )
    outcome_model.fit(X_train, y_out_train)

    home_goal_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("reg", HistGradientBoostingRegressor(max_depth=5, learning_rate=0.05, max_iter=300, random_state=42)),
        ]
    )
    away_goal_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("reg", HistGradientBoostingRegressor(max_depth=5, learning_rate=0.05, max_iter=300, random_state=42)),
        ]
    )
    home_goal_model.fit(X_train, y_h_train)
    away_goal_model.fit(X_train, y_a_train)

    out_pred = outcome_model.predict(X_test)
    out_proba = outcome_model.predict_proba(X_test)
    metrics = {
        "samples": len(train_df),
        "outcome_accuracy": float(accuracy_score(y_out_test, out_pred)),
        "outcome_log_loss": float(log_loss(y_out_test, out_proba, labels=outcome_model.classes_)),
        "home_goals_mae": float(np.mean(np.abs(home_goal_model.predict(X_test) - y_h_test))),
        "away_goals_mae": float(np.mean(np.abs(away_goal_model.predict(X_test) - y_a_test))),
        "classes": outcome_model.named_steps["clf"].classes_.tolist(),
        "feature_columns": feature_cols,
    }

    joblib.dump(outcome_model, MODEL_DIR / "outcome_model.joblib")
    joblib.dump(home_goal_model, MODEL_DIR / "home_goals_model.joblib")
    joblib.dump(away_goal_model, MODEL_DIR / "away_goals_model.joblib")
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()