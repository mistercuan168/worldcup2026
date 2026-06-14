"""World Cup 2026 prediction tool — Streamlit UI (plan §9).

Pick a match → see win/draw/loss odds, the most likely scorelines, both teams'
form, and a plain-English 'why'. A second tab simulates the whole tournament.

Deploy notes (plan §11): on free hosts the .db file may be wiped on restart, so
we rebuild it from the committed CSV on startup (cached for the session).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable when Streamlit runs this file directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.db.models import get_connection
from src.explain.narrative import explain
from src.ingest.build_db import build
from src.model.corners import expected_corners
from src.model.predictor import build_models, predict
from src.simulate.tournament import example_groups, simulate

st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="⚽", layout="wide")


# ---- cached startup: rebuild DB + fit models once per session ----
@st.cache_resource(show_spinner="Building database from committed data…")
def _startup():
    build(verbose=False)  # schema -> historical CSV -> WC fixtures (if API key)
    return True


@st.cache_resource(show_spinner="Training the model (Elo + Dixon-Coles)…")
def _models():
    conn = get_connection()
    bundle = build_models(conn)
    conn.close()
    return bundle


@st.cache_data
def _team_options():
    """Teams with recent data, name -> id, sorted by Elo (strongest first)."""
    bundle = _models()
    conn = get_connection()
    names = {r["team_id"]: r["name"] for r in conn.execute("SELECT team_id,name FROM teams")}
    conn.close()
    teams = [(names[t], t) for t in bundle.dc.fitted_teams if t in names]
    teams.sort(key=lambda nt: bundle.elo_of(nt[1]), reverse=True)
    return teams


_startup()
bundle = _models()
team_opts = _team_options()
name_by_id = {t: n for n, t in team_opts}
id_by_name = {n: t for n, t in team_opts}
names_sorted = [n for n, _ in team_opts]

# ---- sidebar: data freshness + manual refresh (the friend-friendly "live loop") ----
with st.sidebar:
    st.header("Data")
    conn = get_connection()
    played = conn.execute("SELECT COUNT(*) FROM matches WHERE home_goals IS NOT NULL").fetchone()[0]
    latest = conn.execute("SELECT MAX(date) FROM matches WHERE home_goals IS NOT NULL").fetchone()[0]
    conn.close()
    st.metric("Matches in model", f"{played:,}")
    st.caption(f"Latest result: **{latest}**")
    if st.button("🔄 Refresh latest results"):
        # Pull new WC results + retrain. Clears caches so the session rebuilds.
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()
    st.caption("During the tournament, click refresh to pull new results and "
               "re-rate every team.")

st.title("⚽ World Cup 2026 — Match Predictor")
st.caption("Built on ~49k international results since 1872 · Elo + Dixon-Coles · "
           "accuracy measured, not vibes (validation RPS ≈ 0.195).")

tab_match, tab_sim, tab_about = st.tabs(["🆚 Match prediction", "🏆 Tournament odds", "ℹ️ How it works"])

# ============================ MATCH TAB ============================
with tab_match:
    c1, c2, c3 = st.columns([4, 4, 3])
    with c1:
        home = st.selectbox("Team A", names_sorted, index=names_sorted.index("Brazil")
                            if "Brazil" in names_sorted else 0)
    with c2:
        away = st.selectbox("Team B", names_sorted, index=names_sorted.index("Argentina")
                            if "Argentina" in names_sorted else 1)
    with c3:
        neutral = st.checkbox("Neutral venue", value=True,
                              help="Most WC games are neutral. Untick for a host nation at home.")
        knockout = st.checkbox("Knockout match", value=False,
                               help="No draws — ties go to extra time / penalties.")

    if home == away:
        st.warning("Pick two different teams.")
    else:
        conn = get_connection()
        pred = predict(bundle, id_by_name[home], id_by_name[away],
                       neutral=neutral, knockout=knockout, conn=conn)
        conn.close()

        m1, m2, m3 = st.columns(3)
        m1.metric(f"{home} win", f"{pred.p_home:.0%}")
        m2.metric("Draw", f"{pred.p_draw:.0%}")
        m3.metric(f"{away} win", f"{pred.p_away:.0%}")

        if knockout and pred.p_adv_home is not None:
            k1, k2 = st.columns(2)
            k1.metric(f"{home} advances", f"{pred.p_adv_home:.0%}")
            k2.metric(f"{away} advances", f"{pred.p_adv_away:.0%}")

        st.info(explain(pred, home, away))

        left, right = st.columns(2)
        with left:
            st.subheader("Most likely scores")
            ts = pd.DataFrame(
                [{"Score": f"{home} {s} {away}".replace("-", "–"), "Chance": p * 100}
                 for s, p in pred.markets.top_scores],
            )
            st.dataframe(
                ts, hide_index=True, width="stretch",
                column_config={"Chance": st.column_config.ProgressColumn(
                    "Chance", format="%.1f%%", min_value=0, max_value=float(ts["Chance"].max()))},
            )
            o1, o2 = st.columns(2)
            o1.metric("Both teams score", f"{pred.markets.p_btts:.0%}")
            o2.metric("Over 2.5 goals", f"{pred.markets.p_over25:.0%}")
            st.caption(f"Expected goals — {home} {pred.exp_home_goals:.2f} · "
                       f"{away} {pred.exp_away_goals:.2f}")

            ch, ca, ctot = expected_corners(pred.exp_home_goals, pred.exp_away_goals)
            st.caption(f"Estimated corners* — {home} {ch:.1f} · {away} {ca:.1f} · "
                       f"total ~{ctot:.0f}")
            st.caption("\\*Estimate from attacking output, not a trained model "
                       "(no corner data in the dataset).")

        with right:
            st.subheader("Recent form")
            for nm, f in [(home, pred.form_home), (away, pred.form_away)]:
                st.markdown(f"**{nm}** — {f.record}  ·  "
                            f"{' '.join(f.results)}  ·  "
                            f"scoring {f.goals_for:.1f}, conceding {f.goals_against:.1f}")
            st.subheader("Score grid (P of each exact score)")
            n = 6
            grid = pd.DataFrame(
                (pred.matrix[:n, :n] * 100).round(1),
                index=[f"{home} {i}" for i in range(n)],
                columns=[f"{away} {j}" for j in range(n)],
            )
            st.dataframe(grid, width="stretch")

# ========================= SIMULATOR TAB =========================
with tab_sim:
    st.markdown("Simulate the whole tournament thousands of times to get title odds. "
                "Edit the groups below (one team per line, blank line between groups), "
                "then run. *(These illustrative groups are **not** the official 2026 draw — "
                "paste the real groups once it's set.)*")

    default_groups = (
        "Brazil\nSwitzerland\nCameroon\nSerbia\n\n"
        "Argentina\nMexico\nPoland\nSaudi Arabia\n\n"
        "France\nDenmark\nTunisia\nAustralia\n\n"
        "Spain\nGermany\nJapan\nCosta Rica\n\n"
        "England\nNetherlands\nSenegal\nIran\n\n"
        "Portugal\nUruguay\nGhana\nSouth Korea\n\n"
        "Belgium\nCroatia\nMorocco\nCanada\n\n"
        "Colombia\nEcuador\nUnited States\nWales"
    )
    txt = st.text_area("Groups", default_groups, height=240)
    n_sims = st.select_slider("Simulations", [1000, 5000, 10000, 25000], value=10000)

    if st.button("Run simulation", type="primary"):
        blocks = [b for b in txt.split("\n\n") if b.strip()]
        group_names, unknown = [], []
        for b in blocks:
            grp = [ln.strip() for ln in b.splitlines() if ln.strip()]
            for t in grp:
                if t not in id_by_name:
                    unknown.append(t)
            group_names.append(grp)
        if unknown:
            st.error(f"Teams not recognised: {', '.join(sorted(set(unknown)))}")
        else:
            conn = get_connection()
            groups = example_groups(conn, group_names)
            with st.spinner(f"Simulating {n_sims:,} tournaments…"):
                res = simulate(bundle, groups, n_sims=n_sims, conn=conn)
            conn.close()
            rows = [{
                "Team": name_by_id.get(t, str(t)),
                "Win group": p["win_group"] * 100, "Reach SF": p["SF"] * 100,
                "Final": p["F"] * 100, "Win it": p["W"] * 100,
            } for t, p in res.items()]
            df = pd.DataFrame(rows).sort_values("Win it", ascending=False)
            st.dataframe(
                df, hide_index=True, width="stretch",
                column_config={c: st.column_config.ProgressColumn(c, format="%.1f%%",
                                min_value=0, max_value=100.0)
                               for c in ["Win group", "Reach SF", "Final", "Win it"]},
            )

# =========================== ABOUT TAB ===========================
with tab_about:
    st.markdown("""
**What this is.** A World Cup 2026 match predictor trained on ~49,000 international
results going back to 1872 — not just World Cup data (there's too little of that).

**How it predicts.**
- **Elo ratings** rate every nation's strength and carry it forward — great for
  teams that play rarely.
- **Dixon-Coles** (a goals model) turns those strengths into a full grid of exact
  scorelines, from which win/draw/loss, both-teams-to-score and over/under all follow.
- **Recent form** gives a small nudge, and the two models are **blended** (weight
  tuned on past tournaments).

**Is it accurate?** We measure it properly with Ranked Probability Score on
held-out tournaments. On Euro 2024 it scores ≈ 0.195 — in the range of strong
public football models — and it beats simple baselines on World Cup 2022.

**Honest limits.** Single matches are noisy; nobody hits 90%. Smaller nations with
little recent data carry extra uncertainty (flagged in the prediction).
""")
