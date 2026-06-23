"""Streamlit dashboard for the World Cup 2026 Predictor.

Run with:  streamlit run app/streamlit_app.py

Data strategy: tournament odds load from the latest saved snapshot (instant);
Elo + Poisson models are built once and cached for the interactive tabs.
"""

from __future__ import annotations

import copy

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from worldcup.data.teams import display
from worldcup.data.wc2026 import load_groups
from worldcup.features.build import make_feature_row
from worldcup.features.elo import RATINGS_PATH, EloRatingSystem, build_current_ratings
from worldcup.models.poisson import MODEL_PATH as POISSON_PATH
from worldcup.models.poisson import predict_match
from worldcup.simulation.predictions import (
    compare,
    generate_predictions,
    latest_snapshot,
    load_snapshot,
)
from worldcup.simulation.simulator import TournamentSimulator
from worldcup.visualization import plots

STAGE_COLS = ["qualify", "round_of_16", "quarter_final", "semi_final", "final", "champion"]

st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="🏆", layout="wide")
plots.set_style()


# ── Cached loaders ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Preparing data (first run only) ...")
def bootstrap() -> bool:
    """Ensure the historical dataset exists (downloads it on a fresh deploy)."""
    from worldcup.data.download import ensure_results

    ensure_results()
    return True


@st.cache_resource(show_spinner="Loading models ...")
def load_models():
    elo = EloRatingSystem.load() if RATINGS_PATH.exists() else build_current_ratings(save=True)
    if POISSON_PATH.exists():
        from worldcup.models.poisson import load_model

        poisson_bundle = load_model()
    else:
        from worldcup.data.load import load_results
        from worldcup.features.build import build_features
        from worldcup.models.poisson import train_poisson

        poisson_bundle = train_poisson(build_features(load_results()), save=True)
    return elo, poisson_bundle


@st.cache_data(show_spinner="Loading predictions ...")
def load_predictions() -> pd.DataFrame:
    snap = latest_snapshot()
    if snap is not None:
        df, _ = load_snapshot(snap)
        return df
    elo, poisson_bundle = load_models()
    df, _ = generate_predictions(n_sims=10000, elo=elo, poisson_bundle=poisson_bundle, save=True)
    return df


def _pct(df: pd.DataFrame, cols=STAGE_COLS) -> pd.DataFrame:
    """Display copy with probability columns formatted as percentages."""
    out = df.copy()
    for c in cols:
        if c in out:
            out[c] = (out[c] * 100).round(1)
    return out


# ── App ─────────────────────────────────────────────────────────────────────────────
bootstrap()
elo, poisson_bundle = load_models()
groups_df = load_groups()
predictions = load_predictions()
teams = sorted(predictions["team"])

st.title("🏆 World Cup 2026 Predictor")
st.caption("Elo ratings + Poisson scoring + Monte Carlo simulation. Predictions update after every match.")

tab_odds, tab_groups, tab_match, tab_live = st.tabs(
    ["🏆 Title odds", "📊 Groups", "⚔️ Match predictor", "🔄 Live update"]
)

# ── Tab 1: Title odds ─────────────────────────────────────────────────────────────────
with tab_odds:
    st.subheader("Who will win the World Cup?")
    top = predictions.head(3)
    c1, c2, c3 = st.columns(3)
    for col, (_, row) in zip((c1, c2, c3), top.iterrows()):
        col.metric(display(row["team"]), f"{row['champion'] * 100:.1f}%", "to win")

    st.bar_chart(predictions.head(15).set_index("team")["champion"])
    st.dataframe(
        _pct(predictions).rename(columns=lambda c: c.replace("_", " ").title()),
        use_container_width=True,
        hide_index=True,
    )

# ── Tab 2: Groups ─────────────────────────────────────────────────────────────────────
with tab_groups:
    st.subheader("Group qualification odds")
    group_letter = st.selectbox("Select a group", sorted(groups_df["group"].unique()))
    group_teams = groups_df[groups_df["group"] == group_letter]["team"].tolist()
    group_view = predictions[predictions["team"].isin(group_teams)][["team", "qualify", "round_of_16", "champion"]]
    st.dataframe(_pct(group_view, ["qualify", "round_of_16", "champion"]), use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    plots.plot_group_strength(groups_df, elo, ax=ax)
    st.pyplot(fig)

# ── Tab 3: Match predictor ─────────────────────────────────────────────────────────────
with tab_match:
    st.subheader("Head-to-head predictor")
    c1, c2 = st.columns(2)
    home = c1.selectbox("Team A", teams, index=teams.index("Argentina") if "Argentina" in teams else 0)
    away = c2.selectbox("Team B", teams, index=teams.index("Brazil") if "Brazil" in teams else 1)

    if home == away:
        st.warning("Pick two different teams.")
    else:
        row = make_feature_row(home, away, neutral=True, elo=elo)
        result = predict_match(poisson_bundle, row)
        o = result["outcome"]
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{display(home)} win", f"{o['H'] * 100:.1f}%")
        m2.metric("Draw", f"{o['D'] * 100:.1f}%")
        m3.metric(f"{display(away)} win", f"{o['A'] * 100:.1f}%")
        s = result["most_likely_score"]
        st.info(f"Most likely score: **{display(home)} {s['home']}–{s['away']} {display(away)}** ({s['prob']:.1%})")

        fig, ax = plt.subplots(figsize=(6, 5))
        plots.plot_score_matrix(result["matrix"], display(home), display(away), ax=ax)
        st.pyplot(fig)

# ── Tab 4: Live update ─────────────────────────────────────────────────────────────────
with tab_live:
    st.subheader("Apply a finished match and see the impact")
    st.caption("Upload a match CSV (date, home_team, away_team, home_score, away_score, stage). "
               "This preview does not persist — use scripts/update_pipeline.py to save.")
    uploaded = st.file_uploader("Match CSV", type="csv")

    if uploaded is not None:
        from worldcup.data.schema import validate_match_df

        matches = validate_match_df(pd.read_csv(uploaded))
        st.write("Parsed matches:", matches)

        if st.button("Apply & recompute", type="primary"):
            with st.spinner("Re-simulating ..."):
                elo_after = copy.deepcopy(elo)
                for r in matches.itertuples():
                    elo_after.update_match(r.home_team, r.away_team, int(r.home_score), int(r.away_score), neutral=True)
                after = TournamentSimulator(elo_after, poisson_bundle, seed=42).run(5000)
                before = TournamentSimulator(elo, poisson_bundle, seed=42).run(5000)
                movers = compare(before, after, "champion").head(10)
            movers_disp = movers.copy()
            for c in ["champion_before", "champion_after", "delta"]:
                movers_disp[c] = (movers_disp[c] * 100).round(1)
            st.dataframe(movers_disp, use_container_width=True, hide_index=True)
