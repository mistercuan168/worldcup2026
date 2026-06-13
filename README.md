# World Cup 2026 — Match Prediction Tool ⚽

Predicts World Cup 2026 matches — win/draw/loss odds, the most likely scorelines,
both teams' recent form, and a plain-English "why" — plus full-tournament title
odds. Built on ~49,000 international results since 1872, and its accuracy is
**measured**, not guessed (validation RPS ≈ 0.195, in the range of strong public
football models).

![status](https://img.shields.io/badge/status-complete-brightgreen)

## What it does

- **Per-match prediction:** 1X2 probabilities, top exact scores, both-teams-to-score,
  over/under 2.5, expected goals, recent form, and a written explanation.
- **Knockout mode:** redistributes draws into extra-time/penalties → advance odds.
- **Tournament simulator:** Monte Carlo over the whole bracket → "win group / reach
  semis / reach final / win it" for every team.
- **Honest accuracy:** RPS, log-loss, Brier, calibration, and a leak-free backtest
  against past tournaments and baselines.

## Run it locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

streamlit run app/streamlit_app.py     # opens the web app
```

First launch builds the database from the committed CSV and trains the model
(a few seconds), then caches both.

### Handy commands
```bash
python -m src.ingest.build_db          # (re)build the database, print row counts
python -m src.ratings.elo              # top-20 teams by Elo
python -m src.ratings.dixon_coles      # sample matchup predictions
python -m src.simulate.tournament      # 10k-tournament demo
python -m src.evaluate.backtest        # leak-free backtest vs baselines
python -m src.evaluate.tune            # tune blend weight / decay
pytest -q                              # test suite
```

## How it works (the model)

| Layer | Role |
|---|---|
| **World Football Elo** | Rates every nation; carries strength forward (great for teams that rarely play). Baseline 1X2 + a prior for thin-data teams. |
| **Dixon-Coles Poisson** | Time-decayed goals model → a full grid of exact scores. Every market (1X2, BTTS, O/U, top scores) is derived from this one matrix. |
| **Form** | Last ~6 matches give a small nudge to expected goals + display material. |
| **Blend** | Final 1X2 = `w·DixonColes + (1−w)·Elo`, with `w` tuned on a past tournament. |
| **Monte Carlo** | Rolls per-match odds up into tournament odds. |

See `worldcup2026-prediction-plan.md` for the full design rationale.

## Accuracy (measured, not vibed)

- **Euro 2024 (validation):** RPS ≈ **0.195** — strong-model range.
- **World Cup 2022 (test):** beats the naive base-rate baseline and edges Elo-only,
  on an upset-heavy tournament.
- Metric: **Ranked Probability Score** (lower = better; the standard for 1X2).
  Bookmaker-odds comparison is wired but needs an odds dataset to activate.

## Deploying for friends

Free, private, no install for them → see **[DEPLOY.md](DEPLOY.md)** (plain-English).
Short version: push to GitHub → deploy on Streamlit Community Cloud →
`app/streamlit_app.py` → add friends' emails to the viewer allow-list.

## API keys (both optional)

The model trains entirely on the committed `data/results.csv` — **no keys needed**.
Keys only add live WC2026 fixtures/results:
- **football-data.org** — free, 10 req/min
- **API-Football** — free, 100 req/day (its own predictions are used as a
  comparison column only, never as training truth)

Put them in `.env` locally (see `.env.example`) or Streamlit Secrets when deployed.

## Project layout

```
worldcup2026/
├── app/streamlit_app.py    # the web UI
├── data/                   # results.csv + shootouts.csv (committed); worldcup.db is rebuilt
├── src/
│   ├── ingest/             # csv_loader, cached/rate-limited API clients, build_db
│   ├── ratings/            # elo, dixon_coles, form
│   ├── model/              # predictor (blend), knockout, score_matrix (outputs)
│   ├── simulate/           # tournament monte carlo
│   ├── explain/            # plain-English narrative
│   ├── evaluate/           # metrics, backtest, tune
│   └── db/                 # schema.sql, models.py
└── tests/
```
