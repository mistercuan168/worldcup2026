"""Phase 3-6 acceptance: matrix is a valid distribution, outputs are consistent,
and the narrative's numbers match the model exactly."""
import numpy as np

from src.db.models import get_connection
from src.model import score_matrix as sm
from src.ratings.dixon_coles import score_matrix_from_lambdas


def test_score_matrix_is_distribution():
    mat = score_matrix_from_lambdas(1.6, 1.1, -0.08)
    assert abs(mat.sum() - 1.0) < 1e-9
    assert (mat >= 0).all()


def test_outputs_consistent_with_matrix():
    mat = score_matrix_from_lambdas(1.6, 1.1, -0.08)
    out = sm.derive_all(mat)
    # 1X2 must sum to 1 and match direct matrix sums.
    assert abs(out.p_home + out.p_draw + out.p_away - 1.0) < 1e-9
    ph, pd_, pa = sm.outcome_probs(mat)
    assert abs(out.p_home - ph) < 1e-12
    # BTTS and Over2.5 are probabilities.
    assert 0 <= out.p_btts <= 1 and 0 <= out.p_over25 <= 1
    # Higher home lambda -> home favored here.
    assert out.p_home > out.p_away


def test_narrative_numbers_match_model():
    from src.model.predictor import build_models, predict
    from src.explain.narrative import explain

    conn = get_connection()
    names = {r["name"]: r["team_id"] for r in conn.execute("SELECT team_id,name FROM teams")}
    bundle = build_models(conn)
    p = predict(bundle, names["Brazil"], names["Argentina"], neutral=True, conn=conn)
    text = explain(p, "Brazil", "Argentina")
    conn.close()

    # The headline percentages in the text must equal the rounded model values.
    assert f"{p.p_home:.0%}" in text
    assert f"{p.p_away:.0%}" in text
    assert p.markets.top_scores[0][0] in text  # most likely score appears
