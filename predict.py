#!/usr/bin/env python3
"""Predict Mexico vs England World Cup Round of 16 match."""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import build_match_features, load_results

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "results.csv"
MODEL_DIR = ROOT / "models"

MATCH = {
    "date": "2026-07-05",
    "home_team": "Mexico",
    "away_team": "England",
    "venue": "Mexico City, Mexico",
    "neutral": False,
    "tournament": "FIFA World Cup",
    "stage": "Round of 16",
}


def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def score_distribution(
    home_lambda: float,
    away_lambda: float,
    home_team: str,
    away_team: str,
    max_goals: int = 6,
) -> pd.DataFrame:
    rows = []
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = poisson_pmf(h, home_lambda) * poisson_pmf(a, away_lambda)
            if h > a:
                result = f"{home_team} win"
            elif h < a:
                result = f"{away_team} win"
            else:
                result = "Draw"
            rows.append({"home_goals": h, "away_goals": a, "probability": p, "result": result})
    df = pd.DataFrame(rows)
    df["probability"] /= df["probability"].sum()
    return df.sort_values("probability", ascending=False).reset_index(drop=True)


def recent_world_cup_form(team: str, n: int = 4) -> list[dict]:
    df = load_results(str(DATA_PATH))
    wc = df[df["tournament"].str.contains("FIFA World Cup", na=False)]
    wc = wc[(wc["home_team"] == team) | (wc["away_team"] == team)]
    wc = wc[wc["date"] >= "2026-06-01"].tail(n)

    rows = []
    for _, m in wc.iterrows():
        if m["home_team"] == team:
            gf, ga, opp, venue = m["home_score"], m["away_score"], m["away_team"], "H"
        else:
            gf, ga, opp, venue = m["away_score"], m["home_score"], m["home_team"], "A"
        rows.append(
            {
                "date": m["date"].strftime("%Y-%m-%d"),
                "opponent": opp,
                "score": f"{gf}-{ga}",
                "venue": venue,
            }
        )
    return rows


def main() -> None:
    home_team = MATCH["home_team"]
    away_team = MATCH["away_team"]

    outcome_model = joblib.load(MODEL_DIR / "outcome_model.joblib")
    home_goal_model = joblib.load(MODEL_DIR / "home_goals_model.joblib")
    away_goal_model = joblib.load(MODEL_DIR / "away_goals_model.joblib")

    with open(MODEL_DIR / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)

    X, team_context = build_match_features(
        str(DATA_PATH),
        home_team=home_team,
        away_team=away_team,
        neutral=MATCH["neutral"],
        tournament=MATCH["tournament"],
        cutoff_date=MATCH["date"],
    )

    base_features = metrics["feature_columns"]
    X_outcome = X[base_features]
    X_goals = X[base_features]

    proba = outcome_model.predict_proba(X_outcome)[0]
    classes = list(outcome_model.named_steps["clf"].classes_)
    prob_map = {label: float(p) for label, p in zip(classes, proba)}

    home_win_ml = prob_map.get("home_win", 0.0)
    draw_ml = prob_map.get("draw", 0.0)
    away_win_ml = prob_map.get("away_win", 0.0)

    home_lambda = max(0.15, float(home_goal_model.predict(X_goals)[0]))
    away_lambda = max(0.15, float(away_goal_model.predict(X_goals)[0]))

    if not MATCH["neutral"]:
        home_lambda *= 1.08
        away_lambda *= 0.95

    score_df = score_distribution(home_lambda, away_lambda, home_team, away_team)
    poisson_home = float(score_df.loc[score_df["result"] == f"{home_team} win", "probability"].sum())
    poisson_draw = float(score_df.loc[score_df["result"] == "Draw", "probability"].sum())
    poisson_away = float(score_df.loc[score_df["result"] == f"{away_team} win", "probability"].sum())

    home_win = 0.55 * home_win_ml + 0.45 * poisson_home
    draw = 0.55 * draw_ml + 0.45 * poisson_draw
    away_win = 0.55 * away_win_ml + 0.45 * poisson_away
    total = home_win + draw + away_win
    home_win /= total
    draw /= total
    away_win /= total

    top_scores = score_df.head(8)
    most_likely = top_scores.iloc[0]
    predicted_score = f"{int(round(home_lambda))}-{int(round(away_lambda))}"
    expected_score = f"{home_lambda:.2f}-{away_lambda:.2f}"

    if home_win >= max(draw, away_win):
        predicted_winner = home_team
    elif away_win >= max(home_win, draw):
        predicted_winner = away_team
    else:
        predicted_winner = "Draw"

    home_form = recent_world_cup_form(home_team)
    away_form = recent_world_cup_form(away_team)

    report = {
        "match": MATCH,
        "predicted_winner": predicted_winner,
        "win_probabilities": {
            home_team: round(home_win * 100, 1),
            "Draw": round(draw * 100, 1),
            away_team: round(away_win * 100, 1),
        },
        "score_prediction": {
            "most_likely_scoreline": f"{int(most_likely['home_goals'])}-{int(most_likely['away_goals'])}",
            "most_likely_score_probability_pct": round(float(most_likely["probability"]) * 100, 1),
            "rounded_expected_goals": predicted_score,
            "expected_goals": expected_score,
            "top_scorelines": [
                {
                    "score": f"{int(r.home_goals)}-{int(r.away_goals)}",
                    "probability_pct": round(float(r.probability) * 100, 1),
                }
                for r in top_scores.itertuples()
            ],
        },
        "team_context": {
            f"{home_team.lower()}_elo": round(team_context["elo"], 1),
            f"{away_team.lower()}_elo": round(team_context["away_elo"], 1),
            f"{home_team.lower()}_recent_wc_form": home_form,
            f"{away_team.lower()}_recent_wc_form": away_form,
            "head_to_head": {
                "matches": int(team_context["h2h_matches"]),
                f"{home_team.lower()}_win_rate_pct": round(team_context["h2h_home_win_rate"] * 100, 1),
                "draw_rate_pct": round(team_context["h2h_draw_rate"] * 100, 1),
            },
        },
        "model_metrics": {
            "training_matches": metrics["samples"],
            "holdout_accuracy": round(metrics["outcome_accuracy"] * 100, 1),
            "holdout_log_loss": round(metrics["outcome_log_loss"], 3),
        },
    }

    print("=" * 60)
    print(f"{home_team.upper()} vs {away_team.upper()} — 2026 FIFA World Cup ({MATCH['stage']})")
    print(f"Date: {MATCH['date']} | Venue: {MATCH['venue']}")
    print("=" * 60)
    print(f"\nPredicted winner: {report['predicted_winner']}")
    print("\nWin probabilities:")
    for team, pct in report["win_probabilities"].items():
        bar = "#" * int(pct / 2)
        print(f"  {team:8s} {pct:5.1f}%  {bar}")

    print("\nScore prediction:")
    sp = report["score_prediction"]
    print(f"  Most likely:  {sp['most_likely_scoreline']} ({sp['most_likely_score_probability_pct']}%)")
    print(f"  Expected xG:  {sp['expected_goals']}")
    print(f"  Rounded:      {sp['rounded_expected_goals']}")
    print("  Top scorelines:")
    for item in sp["top_scorelines"]:
        print(f"    {item['score']:5s}  {item['probability_pct']:4.1f}%")

    print("\n2026 World Cup form:")
    print(f"  {home_team}:", ", ".join(f"{g['score']} vs {g['opponent']}" for g in home_form))
    print(f"  {away_team}:", ", ".join(f"{g['score']} vs {g['opponent']}" for g in away_form))

    h2h = report["team_context"]["head_to_head"]
    print("\nHead-to-head (historical):", h2h["matches"], "matches")
    print(f"  {home_team} win rate: {h2h[f'{home_team.lower()}_win_rate_pct']}%")
    print(f"  Draw rate:            {h2h['draw_rate_pct']}%")

    print("\nModel quality (holdout):")
    print(f"  Accuracy: {report['model_metrics']['holdout_accuracy']}%")
    print(f"  Log loss: {report['model_metrics']['holdout_log_loss']}")

    out_path = ROOT / "prediction_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to {out_path}")


if __name__ == "__main__":
    main()