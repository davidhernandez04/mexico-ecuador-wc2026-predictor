#!/usr/bin/env python3
"""Generate LinkedIn-ready visuals for the Mexico vs Ecuador prediction."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
REPORT = json.loads((ROOT / "prediction_report.json").read_text(encoding="utf-8"))

# LinkedIn carousel: square works on mobile and desktop
W, H = 1080, 1080
DPI = 150

MEX_GREEN = "#006847"
MEX_RED = "#CE1126"
ECU_YELLOW = "#FFD100"
ECU_BLUE = "#034EA2"
INK = "#1a1a2e"
MUTED = "#6b7280"
BG = "#f8fafc"
CARD = "#ffffff"


def save(fig: plt.Figure, name: str) -> Path:
    path = OUT / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def add_footer(ax: plt.Axes, text: str = "Built with Python + scikit-learn | 32K+ international matches") -> None:
    ax.text(0.5, 0.03, text, transform=ax.transAxes, ha="center", va="bottom", fontsize=11, color=MUTED)


def slide_cover() -> Path:
    fig = plt.figure(figsize=(W / DPI, H / DPI), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(mpatches.FancyBboxPatch((0.05, 0.82), 0.9, 0.12, boxstyle="round,pad=0.02", fc=MEX_GREEN, ec="none"))
    ax.text(0.5, 0.88, "2026 FIFA WORLD CUP  •  ROUND OF 32", ha="center", va="center", fontsize=18, color="white", fontweight="bold")

    ax.text(0.5, 0.68, "MEXICO", ha="center", fontsize=54, fontweight="bold", color=MEX_GREEN)
    ax.text(0.5, 0.60, "vs", ha="center", fontsize=28, color=MUTED, fontstyle="italic")
    ax.text(0.5, 0.52, "ECUADOR", ha="center", fontsize=54, fontweight="bold", color=ECU_BLUE)

    ax.text(0.5, 0.42, "June 30, 2026  •  Mexico City", ha="center", fontsize=16, color=MUTED)

    winner = REPORT["predicted_winner"]
    probs = REPORT["win_probabilities"]
    ax.add_patch(mpatches.FancyBboxPatch((0.12, 0.18), 0.76, 0.20, boxstyle="round,pad=0.03", fc=CARD, ec="#e5e7eb", lw=2))
    ax.text(0.5, 0.31, "ML PREDICTION", ha="center", fontsize=14, color=MUTED, fontweight="bold")
    ax.text(0.5, 0.24, f"{winner} to win", ha="center", fontsize=36, fontweight="bold", color=INK)
    ax.text(0.5, 0.19, f"{probs['Mexico']:.1f}% Mexico  •  {probs['Draw']:.1f}% Draw  •  {probs['Ecuador']:.1f}% Ecuador",
            ha="center", fontsize=15, color=MUTED)

    score = REPORT["score_prediction"]
    ax.text(0.5, 0.10, f"Most likely score: {score['most_likely_scoreline']}  |  xG: {score['expected_goals']}",
            ha="center", fontsize=14, color=INK)

    add_footer(ax)
    return save(fig, "slide_1_cover.png")


def slide_win_probability() -> Path:
    fig = plt.figure(figsize=(W / DPI, H / DPI), facecolor=BG)
    ax = fig.add_axes([0.1, 0.15, 0.85, 0.68])
    ax.set_facecolor(BG)

    labels = ["Mexico", "Draw", "Ecuador"]
    values = [REPORT["win_probabilities"][k] for k in labels]
    colors = [MEX_GREEN, "#9ca3af", ECU_BLUE]

    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, height=0.55, edgecolor="white", linewidth=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=22, fontweight="bold")
    ax.set_xlim(0, 80)
    ax.set_xlabel("Win probability (%)", fontsize=14, color=MUTED)
    ax.set_title("Predicted Match Outcome", fontsize=28, fontweight="bold", color=INK, pad=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25)

    for bar, val in zip(bars, values):
        ax.text(val + 1.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%",
                va="center", fontsize=20, fontweight="bold", color=INK)

    ax.text(0.5, 0.92, "Blended ML classifier + Poisson score model", transform=fig.transFigure,
            ha="center", fontsize=13, color=MUTED)
    add_footer(ax)
    return save(fig, "slide_2_win_probability.png")


def slide_scorelines() -> Path:
    fig = plt.figure(figsize=(W / DPI, H / DPI), facecolor=BG)
    ax = fig.add_axes([0.12, 0.15, 0.82, 0.68])
    ax.set_facecolor(BG)

    scores = REPORT["score_prediction"]["top_scorelines"]
    labels = [s["score"] for s in scores]
    values = [s["probability_pct"] for s in scores]

    x = np.arange(len(labels))
    colors = [MEX_GREEN if s.startswith(("1", "2", "3")) and s.endswith("-0") or s.endswith("-1") else
              ECU_BLUE if s.endswith(("1", "2")) and s.startswith("0") else "#94a3b8" for s in labels]

    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=1.5, width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=18, fontweight="bold")
    ax.set_ylabel("Probability (%)", fontsize=14, color=MUTED)
    ax.set_title("Top Predicted Scorelines", fontsize=28, fontweight="bold", color=INK, pad=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis="y", alpha=0.25)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4, f"{val:.1f}%",
                ha="center", fontsize=13, fontweight="bold", color=INK)

    exp = REPORT["score_prediction"]["expected_goals"]
    ax.text(0.5, 0.92, f"Expected goals: {exp}", transform=fig.transFigure, ha="center", fontsize=14, color=MUTED)
    add_footer(ax)
    return save(fig, "slide_3_scorelines.png")


def slide_team_form() -> Path:
    fig = plt.figure(figsize=(W / DPI, H / DPI), facecolor=BG)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.1], hspace=0.35, wspace=0.25,
                  left=0.08, right=0.92, top=0.88, bottom=0.12)

    fig.suptitle("2026 World Cup Group Stage Form", fontsize=28, fontweight="bold", color=INK)

    ctx = REPORT["team_context"]

    # Elo comparison
    ax1 = fig.add_subplot(gs[0, :])
    teams = ["Mexico", "Ecuador"]
    elos = [ctx["mexico_elo"], ctx["ecuador_elo"]]
    cols = [MEX_GREEN, ECU_BLUE]
    bars = ax1.bar(teams, elos, color=cols, width=0.45, edgecolor="white", linewidth=2)
    ax1.set_ylabel("Elo rating", fontsize=13, color=MUTED)
    ax1.set_ylim(1800, 1950)
    ax1.set_title("Team strength (Elo)", fontsize=18, fontweight="bold", pad=10)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    for bar, val in zip(bars, elos):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, f"{val:.0f}",
                 ha="center", fontweight="bold", fontsize=16)

    def form_panel(ax, team_name, form, color):
        ax.set_facecolor(CARD)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02", fc=CARD, ec="#e5e7eb", lw=1.5))
        ax.text(0.5, 0.88, team_name, ha="center", fontsize=20, fontweight="bold", color=color)
        for i, g in enumerate(form):
            y = 0.68 - i * 0.22
            result = g["score"]
            w = result.split("-")[0] > result.split("-")[1] if "-" in result else False
            badge_color = MEX_GREEN if (team_name == "Mexico" and w) or (team_name == "Ecuador" and w) else (
                "#ef4444" if (result.split("-")[0] < result.split("-")[1]) else "#f59e0b")
            ax.text(0.12, y, result, fontsize=22, fontweight="bold", color=badge_color)
            ax.text(0.38, y, f"vs {g['opponent']}", fontsize=14, color=INK, va="center")

    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    form_panel(ax2, "Mexico", ctx["mexico_recent_wc_form"], MEX_GREEN)
    form_panel(ax3, "Ecuador", ctx["ecuador_recent_wc_form"], ECU_BLUE)

    add_footer(fig.add_axes([0, 0, 1, 0.001]))
    return save(fig, "slide_4_team_form.png")


def slide_methodology() -> Path:
    fig = plt.figure(figsize=(W / DPI, H / DPI), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.90, "How the model works", ha="center", fontsize=30, fontweight="bold", color=INK)

    steps = [
        ("1", "Data", "49K+ international matches\n(1872 – 2026)"),
        ("2", "Features", "Elo ratings, rolling form,\nhead-to-head, home advantage"),
        ("3", "Models", "Gradient Boosting (outcome)\n+ goal regressors (xG)"),
        ("4", "Output", "Win %, score distribution,\nmost likely scoreline"),
    ]

    for i, (num, title, desc) in enumerate(steps):
        y = 0.72 - i * 0.18
        ax.add_patch(mpatches.Circle((0.14, y), 0.035, fc=MEX_GREEN, ec="none"))
        ax.text(0.14, y, num, ha="center", va="center", fontsize=16, color="white", fontweight="bold")
        ax.text(0.22, y + 0.03, title, fontsize=20, fontweight="bold", color=INK, va="center")
        ax.text(0.22, y - 0.05, desc, fontsize=13, color=MUTED, va="top")

    metrics = REPORT["model_metrics"]
    h2h = REPORT["team_context"]["head_to_head"]
    ax.add_patch(mpatches.FancyBboxPatch((0.08, 0.08), 0.84, 0.16, boxstyle="round,pad=0.02", fc=CARD, ec="#e5e7eb", lw=1.5))
    ax.text(0.5, 0.18,
            f"Holdout accuracy: {metrics['holdout_accuracy']}%  •  "
            f"Training matches: {metrics['training_matches']:,}  •  "
            f"H2H sample: {h2h['matches']} games",
            ha="center", fontsize=13, color=INK)
    ax.text(0.5, 0.11, "Football is noisy — models inform decisions, they don't guarantee results.",
            ha="center", fontsize=12, color=MUTED, fontstyle="italic")

    add_footer(ax)
    return save(fig, "slide_5_methodology.png")


def slide_single_summary() -> Path:
    """One-image version for posts that don't use a carousel."""
    fig = plt.figure(figsize=(W / DPI, 627 / DPI), facecolor=BG)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.1, 1], wspace=0.25, left=0.06, right=0.96, top=0.82, bottom=0.14)

    fig.suptitle("Mexico vs Ecuador — ML Match Prediction", fontsize=22, fontweight="bold", color=INK)
    fig.text(0.5, 0.90, "2026 FIFA World Cup Round of 32  •  June 30  •  Mexico City", ha="center", fontsize=12, color=MUTED)

    ax1 = fig.add_subplot(gs[0, 0])
    labels = ["Mexico", "Draw", "Ecuador"]
    values = [REPORT["win_probabilities"][k] for k in labels]
    colors = [MEX_GREEN, "#9ca3af", ECU_BLUE]
    ax1.barh(labels, values, color=colors, height=0.55)
    ax1.set_xlim(0, 75)
    ax1.set_title("Win probability", fontweight="bold")
    for i, v in enumerate(values):
        ax1.text(v + 1, i, f"{v:.1f}%", va="center", fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2 = fig.add_subplot(gs[0, 1])
    scores = REPORT["score_prediction"]["top_scorelines"][:5]
    ax2.bar([s["score"] for s in scores], [s["probability_pct"] for s in scores], color=MEX_GREEN, alpha=0.85)
    ax2.set_title("Top scorelines", fontweight="bold")
    ax2.set_ylabel("%")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    sp = REPORT["score_prediction"]
    fig.text(0.5, 0.06, f"Predicted winner: {REPORT['predicted_winner']}  |  "
             f"Most likely: {sp['most_likely_scoreline']}  |  xG: {sp['expected_goals']}",
             ha="center", fontsize=11, color=MUTED)

    return save(fig, "linkedin_single_landscape.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [
        slide_cover(),
        slide_win_probability(),
        slide_scorelines(),
        slide_team_form(),
        slide_methodology(),
        slide_single_summary(),
    ]
    print("Created LinkedIn visuals:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()