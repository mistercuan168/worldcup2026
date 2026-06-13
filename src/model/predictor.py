"""The predictor: blends Elo + Dixon-Coles into one prediction (plan §6d).

Final 1X2 = w · DixonColes + (1 − w) · Elo.
Elo stabilises thin-data teams (cold start); DC supplies the scoreline matrix.
A small form nudge adjusts expected goals. `w` is a default here and is tuned on
the backtest in Phase 8.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np

from src.db.models import get_connection
from src.model import score_matrix as sm
from src.model.knockout import resolve_knockout
from src.ratings.dixon_coles import (DixonColesModel, fit_dixon_coles,
                                      score_matrix_from_lambdas)
from src.ratings.elo import EloModel, build_elo, elo_1x2
from src.ratings.form import Form, get_form

MODEL_VERSION = "v1-elo+dc-blend"
DEFAULT_BLEND_W = 0.75   # weight on Dixon-Coles. Tuned on Euro 2024 (val RPS 0.195);
                         # kept just below the 0.80 optimum to avoid overfitting one
                         # tournament and to retain Elo stabilization for thin-data teams.
FORM_W = 0.18            # how hard recent form nudges expected goals (small)


@dataclass
class ModelBundle:
    elo: EloModel
    dc: DixonColesModel
    as_of: str | None
    blend_w: float = DEFAULT_BLEND_W

    def elo_of(self, team_id: int) -> float:
        # Use the snapshot as-of date when backtesting; else current rating.
        return self.elo.get(team_id)


def build_models(
    conn: sqlite3.Connection | None = None,
    as_of: str | None = None,
    blend_w: float = DEFAULT_BLEND_W,
    **dc_kwargs,
) -> ModelBundle:
    own = conn is None
    if own:
        conn = get_connection()
    elo = build_elo(conn, snapshot=(as_of is None), as_of=as_of)
    dc = fit_dixon_coles(conn, as_of=as_of, **dc_kwargs)
    if own:
        conn.close()
    return ModelBundle(elo=elo, dc=dc, as_of=as_of, blend_w=blend_w)


@dataclass
class Prediction:
    home_id: int
    away_id: int
    neutral: bool
    # blended 1X2
    p_home: float
    p_draw: float
    p_away: float
    markets: sm.MarketOutputs
    matrix: np.ndarray
    # internals for the 'why' explanation
    elo_home: float
    elo_away: float
    dc_attack_home: float
    dc_attack_away: float
    dc_defense_home: float
    dc_defense_away: float
    exp_home_goals: float
    exp_away_goals: float
    form_home: Form
    form_away: Form
    cold_start: bool
    # optional knockout advancement
    p_adv_home: float | None = None
    p_adv_away: float | None = None


def predict(
    bundle: ModelBundle,
    home_id: int,
    away_id: int,
    neutral: bool = True,
    knockout: bool = False,
    conn: sqlite3.Connection | None = None,
) -> Prediction:
    own = conn is None
    if own:
        conn = get_connection()

    # --- expected goals from DC, nudged by recent form ---
    lam, mu = bundle.dc.expected_goals(home_id, away_id, neutral)
    fh = get_form(home_id, as_of=bundle.as_of, conn=conn)
    fa = get_form(away_id, as_of=bundle.as_of, conn=conn)
    lam *= 1.0 + FORM_W * (fh.points_rate - 0.5)
    mu *= 1.0 + FORM_W * (fa.points_rate - 0.5)
    matrix = score_matrix_from_lambdas(lam, mu, bundle.dc.rho)
    markets = sm.derive_all(matrix)

    # --- blend 1X2: DC matrix vs Elo baseline ---
    dc_h, dc_d, dc_a = markets.p_home, markets.p_draw, markets.p_away
    eh, ea = bundle.elo_of(home_id), bundle.elo_of(away_id)
    el_h, el_d, el_a = elo_1x2(eh, ea, neutral)
    w = bundle.blend_w
    p_home = w * dc_h + (1 - w) * el_h
    p_draw = w * dc_d + (1 - w) * el_d
    p_away = w * dc_a + (1 - w) * el_a
    s = p_home + p_draw + p_away
    p_home, p_draw, p_away = p_home / s, p_draw / s, p_away / s

    cold = (home_id not in bundle.dc.fitted_teams) or (away_id not in bundle.dc.fitted_teams)

    pred = Prediction(
        home_id=home_id, away_id=away_id, neutral=neutral,
        p_home=p_home, p_draw=p_draw, p_away=p_away,
        markets=markets, matrix=matrix,
        elo_home=eh, elo_away=ea,
        dc_attack_home=bundle.dc.attack.get(home_id, 0.0),
        dc_attack_away=bundle.dc.attack.get(away_id, 0.0),
        dc_defense_home=bundle.dc.defense.get(home_id, 0.0),
        dc_defense_away=bundle.dc.defense.get(away_id, 0.0),
        exp_home_goals=lam, exp_away_goals=mu,
        form_home=fh, form_away=fa,
        cold_start=cold,
    )
    if knockout:
        pred.p_adv_home, pred.p_adv_away = resolve_knockout(
            p_home, p_draw, p_away, eh, ea
        )
    if own:
        conn.close()
    return pred
