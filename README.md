# World Cup 2026 Match Predictor

Machine learning pipeline that predicts the winner, win probabilities, and score distribution for FIFA World Cup 2026 knockout matches.

## Prediction — Mexico vs England (July 5, 2026)

| Outcome | Probability |
|---------|-------------|
| **Mexico win** | **48.0%** |
| Draw | 22.0% |
| England win | 30.0% |

- **Predicted winner:** Mexico
- **Most likely score:** 1-1 (13.0%)
- **Second most likely:** 1-0 (12.6%)
- **Expected goals:** 1.30 – 1.03

### Previous prediction — Mexico vs Ecuador (June 30, 2026)

The model's second-most-likely scoreline (2-0, 12.7%) was the actual result.

| Outcome | Probability |
|---------|-------------|
| Mexico win | 66.7% |
| Draw | 17.4% |
| Ecuador win | 15.9% |

- **Most likely score:** 1-0 (17.5%)
- **Actual score:** 2-0

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

1. **Data** — 49,000+ international matches from [martj42/international_results](https://github.com/martj42/international_results), updated with 2026 World Cup results through July 4
2. **Features** — Elo ratings, 10-game rolling form, head-to-head history, home/neutral flag, tournament weighting
3. **Models**
   - `GradientBoostingClassifier` — match outcome (home win / draw / away win)
   - `HistGradientBoostingRegressor` — expected home and away goals
   - Poisson distribution — scoreline probabilities
4. **Blend** — classifier + Poisson outcome probabilities for final win %

**Holdout performance:** 57.5% outcome accuracy on 32K+ training matches.

## Project structure

```
├── features.py          # Elo + rolling form feature engineering
├── train.py             # Train and save models
├── predict.py           # Run prediction for Mexico vs England
├── prediction_report.json
├── data/results.csv     # International match history
└── models/              # Trained model artifacts
```

## Disclaimer

This is probabilistic forecasting, not a guarantee. Football has high variance — the model quantifies uncertainty; it does not eliminate it.

## License

MIT