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
- [ ] **Phase 2** — Historical football data collection
- [ ] **Phase 3** — World Cup 2026 data collection
- [ ] **Phase 4** — Elo rating system
- [ ] **Phase 5** — Exploratory data analysis
- [ ] **Phase 6** — Feature engineering
- [ ] **Phase 7** — Baseline prediction model
- [ ] **Phase 8** — XGBoost model
- [ ] **Phase 9** — Poisson score model
- [ ] **Phase 10** — Monte Carlo tournament simulator
- [ ] **Phase 11** — Qualification & championship probabilities
- [ ] **Phase 12** — Streamlit dashboard
- [ ] **Phase 13** — Deployment
- [ ] **Phase 14** — Model improvement

---

## License

MIT © Achraf Berbaoui
