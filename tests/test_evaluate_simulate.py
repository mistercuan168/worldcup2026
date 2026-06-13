"""Phase 7-8 acceptance: metrics behave, simulator is internally consistent."""
import math

from src.db.models import get_connection
from src.evaluate.metrics import (brier_score, log_loss,
                                   ranked_probability_score)
from src.model.predictor import build_models
from src.simulate.tournament import example_groups, simulate


def test_rps_perfect_and_worst():
    # Perfect, confident, correct prediction -> 0.
    assert ranked_probability_score((1.0, 0.0, 0.0), "home") == 0.0
    # Confident but wrong (predict home, away happens) -> the maximum, 1.0.
    assert abs(ranked_probability_score((1.0, 0.0, 0.0), "away") - 1.0) < 1e-9
    # Closer-but-wrong (draw) scores better than far-wrong (away).
    near = ranked_probability_score((1.0, 0.0, 0.0), "draw")
    far = ranked_probability_score((1.0, 0.0, 0.0), "away")
    assert near < far


def test_logloss_and_brier():
    assert log_loss((0.5, 0.3, 0.2), "home") == -math.log(0.5)
    assert abs(brier_score((1.0, 0.0, 0.0), "home")) < 1e-12


def test_simulator_consistency():
    conn = get_connection()
    bundle = build_models(conn)
    groups = example_groups(conn, [
        ["Brazil", "Serbia", "Switzerland", "Cameroon"],
        ["Argentina", "Mexico", "Poland", "Saudi Arabia"],
    ])
    res = simulate(bundle, groups, n_sims=1000, conn=conn, seed=1)
    conn.close()

    for t, p in res.items():
        # advancement is monotonic deep into the bracket
        assert p["SF"] >= p["F"] - 1e-9
        assert p["F"] >= p["W"] - 1e-9
        # all probabilities are valid
        assert all(0.0 <= v <= 1.0 for v in p.values())

    # exactly one champion's worth of probability mass
    total_win = sum(p["W"] for p in res.values())
    assert abs(total_win - 1.0) < 1e-9
