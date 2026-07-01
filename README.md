# Mexico vs Ecuador — World Cup 2026 Match Predictor

Machine learning pipeline that predicts the winner, win probabilities, and score distribution for the **Mexico vs Ecuador** FIFA World Cup Round of 32 match (June 30, 2026, Mexico City).

## Prediction (as of June 30, 2026)

| Outcome | Probability |
|---------|-------------|
| **Mexico win** | **66.7%** |
| Draw | 17.4% |
| Ecuador win | 15.9% |

- **Predicted winner:** Mexico
- **Most likely score:** 1-0 (17.5%)
- **Expected goals:** 1.44 – 0.67

## Quick start

```bash
git clone https://github.com/davidhernandez04/mexico-ecuador-wc2026-predictor.git
cd mexico-ecuador-wc2026-predictor
pip install -r requirements.txt
python predict.py
```

To retrain from scratch:

```bash
python train.py
python predict.py
```

## How it works

1. **Data** — 49,000+ international matches from [martj42/international_results](https://github.com/martj42/international_results)
2. **Features** — Elo ratings, 10-game rolling form, head-to-head history, home/neutral flag, tournament weighting
3. **Models**
   - `GradientBoostingClassifier` — match outcome (home win / draw / away win)
   - `HistGradientBoostingRegressor` — expected home and away goals
   - Poisson distribution — scoreline probabilities
4. **Blend** — classifier + Poisson outcome probabilities for final win %

**Holdout performance:** 57.2% outcome accuracy on 32K+ training matches.

## Project structure

```
├── features.py          # Elo + rolling form feature engineering
├── train.py             # Train and save models
├── predict.py           # Run prediction for Mexico vs Ecuador
├── prediction_report.json
├── data/results.csv     # International match history
├── models/              # Trained model artifacts
└── linkedin/            # Carousel visuals + post copy for sharing
```

## LinkedIn assets

Pre-built carousel slides and post text live in `linkedin/`. Regenerate visuals:

```bash
python linkedin/create_visuals.py
```

## Disclaimer

This is probabilistic forecasting, not a guarantee. Football has high variance — the model quantifies uncertainty; it does not eliminate it.

## License

MIT