"""Feature engineering for international football match prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

ELO_K = 32
ELO_HOME_ADV = 65
ELO_INITIAL = 1500
ROLLING_WINDOW = 10

TOURNAMENT_WEIGHT = {
    "FIFA World Cup": 1.0,
    "FIFA World Cup qualification": 0.85,
    "Copa América": 0.9,
    "UEFA Euro": 0.9,
    "AFC Asian Cup": 0.85,
    "Africa Cup of Nations": 0.85,
    "CONCACAF Gold Cup": 0.8,
    "Friendly": 0.6,
}


def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def tournament_importance(tournament: str) -> float:
    if tournament in TOURNAMENT_WEIGHT:
        return TOURNAMENT_WEIGHT[tournament]
    name = tournament.lower()
    if "world cup" in name:
        return 0.95
    if "friendly" in name:
        return 0.6
    if "qualification" in name or "qualifying" in name:
        return 0.8
    return 0.75


def expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def update_elo(
    home_elo: float,
    away_elo: float,
    home_score: int,
    away_score: int,
    neutral: bool,
    tournament: str,
) -> tuple[float, float]:
    weight = tournament_importance(tournament)
    home_adj = home_elo + (0 if neutral else ELO_HOME_ADV)
    exp_home = expected_score(home_adj, away_elo)

    if home_score > away_score:
        actual_home = 1.0
    elif home_score < away_score:
        actual_home = 0.0
    else:
        actual_home = 0.5

    margin = abs(home_score - away_score)
    multiplier = np.log(max(margin, 1) + 1) * (2.2 / (0.001 * (home_adj - away_elo) + 2.2))
    delta = ELO_K * weight * multiplier * (actual_home - exp_home)

    return home_elo + delta, away_elo - delta


class TeamState:
    def __init__(self) -> None:
        self.elo = ELO_INITIAL
        self.recent_goals_for: list[int] = []
        self.recent_goals_against: list[int] = []
        self.recent_points: list[float] = []
        self.recent_wins: list[int] = []

    def snapshot(self) -> dict[str, float]:
        goals_for = self.recent_goals_for[-ROLLING_WINDOW:]
        goals_against = self.recent_goals_against[-ROLLING_WINDOW:]
        points = self.recent_points[-ROLLING_WINDOW:]
        wins = self.recent_wins[-ROLLING_WINDOW:]

        n = max(len(goals_for), 1)
        return {
            "elo": self.elo,
            "avg_goals_for": np.mean(goals_for) if goals_for else 1.2,
            "avg_goals_against": np.mean(goals_against) if goals_against else 1.2,
            "avg_points": np.mean(points) if points else 1.0,
            "win_rate": np.mean(wins) if wins else 0.33,
            "matches_played": len(self.recent_goals_for),
        }

    def update(self, goals_for: int, goals_against: int) -> None:
        if goals_for > goals_against:
            points = 3.0
            win = 1
        elif goals_for == goals_against:
            points = 1.0
            win = 0
        else:
            points = 0.0
            win = 0

        self.recent_goals_for.append(goals_for)
        self.recent_goals_against.append(goals_against)
        self.recent_points.append(points)
        self.recent_wins.append(win)


def build_training_frame(df: pd.DataFrame, min_year: int = 1990) -> pd.DataFrame:
    df = df[df["date"].dt.year >= min_year].copy()
    teams: dict[str, TeamState] = {}
    rows: list[dict] = []

    for _, match in df.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        teams.setdefault(home, TeamState())
        teams.setdefault(away, TeamState())

        home_state = teams[home].snapshot()
        away_state = teams[away].snapshot()

        home_score = int(match["home_score"])
        away_score = int(match["away_score"])
        if home_score > away_score:
            outcome = "home_win"
        elif home_score < away_score:
            outcome = "away_win"
        else:
            outcome = "draw"

        rows.append(
            {
                "date": match["date"],
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
                "outcome": outcome,
                "neutral": bool(match["neutral"]),
                "tournament": match["tournament"],
                "is_world_cup": float("FIFA World Cup" in str(match["tournament"])),
                "home_elo": home_state["elo"],
                "away_elo": away_state["elo"],
                "elo_diff": home_state["elo"] - away_state["elo"],
                "home_avg_goals_for": home_state["avg_goals_for"],
                "home_avg_goals_against": home_state["avg_goals_against"],
                "home_avg_points": home_state["avg_points"],
                "home_win_rate": home_state["win_rate"],
                "home_matches_played": home_state["matches_played"],
                "away_avg_goals_for": away_state["avg_goals_for"],
                "away_avg_goals_against": away_state["avg_goals_against"],
                "away_avg_points": away_state["avg_points"],
                "away_win_rate": away_state["win_rate"],
                "away_matches_played": away_state["matches_played"],
                "attack_diff": home_state["avg_goals_for"] - away_state["avg_goals_for"],
                "defense_diff": away_state["avg_goals_against"] - home_state["avg_goals_against"],
                "form_diff": home_state["avg_points"] - away_state["avg_points"],
            }
        )

        teams[home].update(home_score, away_score)
        teams[away].update(away_score, home_score)
        h_elo, a_elo = update_elo(
            teams[home].elo,
            teams[away].elo,
            home_score,
            away_score,
            bool(match["neutral"]),
            str(match["tournament"]),
        )
        teams[home].elo = h_elo
        teams[away].elo = a_elo

    return pd.DataFrame(rows)


FEATURE_COLUMNS = [
    "neutral",
    "is_world_cup",
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_avg_goals_for",
    "home_avg_goals_against",
    "home_avg_points",
    "home_win_rate",
    "home_matches_played",
    "away_avg_goals_for",
    "away_avg_goals_against",
    "away_avg_points",
    "away_win_rate",
    "away_matches_played",
    "attack_diff",
    "defense_diff",
    "form_diff",
]


def build_match_features(
    results_path: str,
    home_team: str,
    away_team: str,
    neutral: bool,
    tournament: str,
    cutoff_date: str | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    df = load_results(results_path)
    if cutoff_date:
        df = df[df["date"] < pd.Timestamp(cutoff_date)]

    teams: dict[str, TeamState] = {}
    h2h_home_wins = 0
    h2h_away_wins = 0
    h2h_draws = 0
    h2h_home_goals = 0
    h2h_away_goals = 0
    h2h_matches = 0

    for _, match in df.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        teams.setdefault(home, TeamState())
        teams.setdefault(away, TeamState())

        home_score = int(match["home_score"])
        away_score = int(match["away_score"])

        pair = {home, away}
        if pair == {home_team, away_team}:
            h2h_matches += 1
            if home == home_team:
                h2h_home_goals += home_score
                h2h_away_goals += away_score
                if home_score > away_score:
                    h2h_home_wins += 1
                elif home_score < away_score:
                    h2h_away_wins += 1
                else:
                    h2h_draws += 1
            else:
                h2h_home_goals += away_score
                h2h_away_goals += home_score
                if away_score > home_score:
                    h2h_home_wins += 1
                elif away_score < home_score:
                    h2h_away_wins += 1
                else:
                    h2h_draws += 1

        teams[home].update(home_score, away_score)
        teams[away].update(away_score, home_score)
        h_elo, a_elo = update_elo(
            teams[home].elo,
            teams[away].elo,
            home_score,
            away_score,
            bool(match["neutral"]),
            str(match["tournament"]),
        )
        teams[home].elo = h_elo
        teams[away].elo = a_elo

    teams.setdefault(home_team, TeamState())
    teams.setdefault(away_team, TeamState())
    home_state = teams[home_team].snapshot()
    away_state = teams[away_team].snapshot()

    h2h = {
        "h2h_matches": h2h_matches,
        "h2h_home_win_rate": h2h_home_wins / h2h_matches if h2h_matches else 0.33,
        "h2h_draw_rate": h2h_draws / h2h_matches if h2h_matches else 0.33,
        "h2h_home_avg_goals": h2h_home_goals / h2h_matches if h2h_matches else 1.2,
        "h2h_away_avg_goals": h2h_away_goals / h2h_matches if h2h_matches else 1.2,
    }

    row = {
        "neutral": float(neutral),
        "is_world_cup": float("FIFA World Cup" in tournament),
        "home_elo": home_state["elo"],
        "away_elo": away_state["elo"],
        "elo_diff": home_state["elo"] - away_state["elo"],
        "home_avg_goals_for": home_state["avg_goals_for"],
        "home_avg_goals_against": home_state["avg_goals_against"],
        "home_avg_points": home_state["avg_points"],
        "home_win_rate": home_state["win_rate"],
        "home_matches_played": home_state["matches_played"],
        "away_avg_goals_for": away_state["avg_goals_for"],
        "away_avg_goals_against": away_state["avg_goals_against"],
        "away_avg_points": away_state["avg_points"],
        "away_win_rate": away_state["win_rate"],
        "away_matches_played": away_state["matches_played"],
        "attack_diff": home_state["avg_goals_for"] - away_state["avg_goals_for"],
        "defense_diff": away_state["avg_goals_against"] - home_state["avg_goals_against"],
        "form_diff": home_state["avg_points"] - away_state["avg_points"],
        **h2h,
    }

    feature_cols = FEATURE_COLUMNS + [
        "h2h_matches",
        "h2h_home_win_rate",
        "h2h_draw_rate",
        "h2h_home_avg_goals",
        "h2h_away_avg_goals",
    ]
    return pd.DataFrame([row])[feature_cols], {**home_state, **{f"away_{k}": v for k, v in away_state.items()}, **h2h}