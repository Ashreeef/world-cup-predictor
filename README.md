# 🏆 World Cup Predictor

A **dynamic** machine-learning system that predicts FIFA 2026 World Cup outcomes — and
**updates its predictions after every match** as the tournament unfolds.

> Example: *Before the group stage, Argentina's title probability is 16%. After
> beating Germany, the system recomputes it to 21% — automatically.*

![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-in%20development-orange)

---

## What it predicts

- ✅ Match outcomes (win / draw / loss)
- ✅ Score-line probabilities
- ✅ Group qualification probabilities
- ✅ Knockout-stage progression probabilities
- ✅ World Cup winner probabilities

## How it works (high level)

| Layer | Technique | Role |
|-------|-----------|------|
| Ratings | **Elo** | Continuously-updated team strength |
| Match model | **XGBoost / Logistic Regression** | P(win/draw/loss) from features |
| Score model | **Poisson regression** | Score-line probabilities |
| Tournament | **Monte Carlo simulation** | Simulates the bracket thousands of times |
| Live loop | **Incremental update pipeline** | Refreshes everything after each match — *no full retrain* |

## The dynamic update loop

```
data/live/match_updates/new_match.csv
            │
            ▼
   scripts/update_pipeline.py
            │  ├─ update Elo + form (incremental)
            │  ├─ update group standings
            │  └─ re-run Monte Carlo simulation
            ▼
   data/predictions/  ──►  reports/ (before vs after)  ──►  dashboard refreshes
```

---

## Project structure

```
world-cup-predictor/
├── data/
│   ├── raw/                 # immutable source data
│   ├── processed/           # cleaned, model-ready data
│   ├── live/match_updates/  # incoming finished-match CSVs
│   └── predictions/         # timestamped prediction snapshots
├── notebooks/               # exploratory analysis
├── src/worldcup/            # the installable package (all real code)
│   ├── config.py            # central path configuration
│   ├── data/                # loading / cleaning / acquisition
│   ├── features/            # Elo, form, rankings
│   ├── models/              # baseline, XGBoost, Poisson
│   ├── simulation/          # Monte Carlo simulator
│   └── visualization/       # plots & reports
├── scripts/                 # entry points (e.g. update_pipeline.py)
├── app/                     # Streamlit dashboard
├── artifacts/               # trained model files (git-ignored)
├── reports/                 # before/after comparison reports
├── tests/                   # pytest suite
├── requirements.txt
├── pyproject.toml
├── init_project.py
└── README.md
```

---

## Getting started

```bash
# 1. Clone
git clone https://github.com/Ashreeef/world-cup-predictor.git
cd world-cup-predictor

# 2. Create & activate a virtual environment (Python 3.11)
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate

# 3. Install dependencies + the package (editable mode)
pip install -r requirements.txt
pip install -e .

# 4. Create the data/ folder structure
python init_project.py

# 5. Verify everything works
pytest
```

### Common commands

| Task | Command |
|------|---------|
| Install dependencies + package | `pip install -r requirements.txt && pip install -e .` |
| Create directory structure | `python init_project.py` |
| Format & lint | `black src scripts tests && isort src scripts tests && flake8 src scripts tests` |
| Run the test suite | `pytest` |
| Apply a finished match | `python scripts/update_pipeline.py --match data/live/match_updates/match_001.csv` |
| Launch the dashboard | `streamlit run app/streamlit_app.py` |

---

## Roadmap

- [x] **Phase 1** — Repository, structure & environment setup
- [x] **Phase 2** — Historical football data collection
- [x] **Phase 3** — World Cup 2026 data collection
- [x] **Phase 4** — Elo rating system
- [x] **Phase 5** — Exploratory data analysis
- [x] **Phase 6** — Feature engineering
- [x] **Phase 7** — Baseline prediction model
- [x] **Phase 8** — XGBoost model
- [x] **Phase 9** — Poisson score model
- [x] **Phase 10** — Monte Carlo tournament simulator
- [x] **Phase 11** — Qualification & championship probabilities
- [x] **Phase 12** — Streamlit dashboard
- [x] **Phase 13** — Deployment
- [x] **Phase 14** — Model improvement (tournament-weighted Elo, calibration, fixed bracket, Dixon-Coles)

---

## Model performance

Evaluated with a time-based split (train < 2022, test ≥ 2022, incl. WC2026):

| Model | Accuracy | Log-loss |
|-------|---------:|---------:|
| Always-home baseline | 0.478 | 1.099 |
| Logistic regression | 0.601 | 0.871 |
| XGBoost | 0.602 | 0.873 |
| Poisson (W/D/L) | 0.603 | 0.877 |

The three trained models tie — with only 4 features and `elo_diff` dominating
(~75% of importance), the signal is largely linear, so model complexity adds
little. Bigger gains will come from richer features (Phase 14), not fancier
models. The Poisson model additionally produces full scoreline probabilities.

Probabilities are well-calibrated (multiclass **Brier 0.514** vs 0.667 uniform).
A **Dixon-Coles** model improves exact-score likelihood (−2.74 vs −2.86 for the
Elo-Poisson) and is provided as a standalone comparison.

Sample title odds (10,000 simulations, fixed bracket): Argentina **25.4%**,
Spain **23.0%**, France **11.5%**, England **8.7%**, Brazil **4.5%**.

---

## Deployment

The dashboard deploys to **Streamlit Community Cloud** with zero committed data:
on first run it downloads the dataset and builds the models + a prediction
snapshot (cached thereafter).

1. Push this repo to GitHub.
2. At [share.streamlit.io](https://share.streamlit.io), create an app pointing
   to `app/streamlit_app.py` on the `main` branch.
3. Streamlit installs `requirements.txt` (which includes `-e .`, so the
   `worldcup` package is importable) and launches the app.

Cold start takes ~30 s while it bootstraps; subsequent loads are instant.

---

## License

MIT © Achraf Berbaoui
