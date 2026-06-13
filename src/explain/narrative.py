"""Plain-English 'why' generator (plan §7).

Template-driven, built only from the model's own internals. Every number in the
text is a real model value — no invented stats. It names the top 2-3 drivers
(Elo gap, attack/defense mismatch, form), states the headline + most likely score,
and is honest about uncertainty.
"""
from __future__ import annotations

from src.model.predictor import Prediction


def _confidence_phrase(p_top: float, margin: float) -> str:
    if p_top >= 0.60:
        return "a clear favorite"
    if margin < 0.08:
        return "essentially a coin-flip"
    if margin < 0.18:
        return "a close match"
    return "a modest favorite"


def explain(pred: Prediction, home_name: str, away_name: str) -> str:
    ph, pd_, pa = pred.p_home, pred.p_draw, pred.p_away
    top_score, top_p = pred.markets.top_scores[0]

    # Identify favorite + how clear it is.
    probs = {home_name: ph, away_name: pa}
    fav = max(probs, key=probs.get)
    fav_p = probs[fav]
    margin = abs(ph - pa)
    conf = _confidence_phrase(fav_p, margin)

    # --- collect drivers as (text, team_it_favors), strongest first ---
    drivers: list[tuple[str, str]] = []

    elo_gap = pred.elo_home - pred.elo_away
    if abs(elo_gap) >= 40:
        stronger = home_name if elo_gap > 0 else away_name
        drivers.append((f"a {abs(elo_gap):.0f}-point Elo edge", stronger))

    # Attack/defense mismatch (DC strengths). Higher attack = scores more;
    # higher defense = concedes fewer.
    atk_gap = pred.dc_attack_home - pred.dc_attack_away
    if abs(atk_gap) >= 0.20:
        better = home_name if atk_gap > 0 else away_name
        drivers.append(("a stronger attack", better))
    def_gap = pred.dc_defense_home - pred.dc_defense_away
    if abs(def_gap) >= 0.20:
        better = home_name if def_gap > 0 else away_name
        drivers.append(("a tighter defense", better))

    # Form
    fh, fa = pred.form_home, pred.form_away
    if fh.n and fa.n and abs(fh.points_rate - fa.points_rate) >= 0.20:
        hotter = home_name if fh.points_rate > fa.points_rate else away_name
        drivers.append(("better recent form", hotter))

    if not pred.neutral:
        drivers.append(("home advantage", home_name))

    # Split drivers by whether they support the favorite or cut against it.
    supporting = [f"{txt} for {team}" for txt, team in drivers if team == fav]
    opposing = [f"{txt} for {team}" for txt, team in drivers if team != fav]

    # --- assemble sentences ---
    headline = (
        f"**{home_name} {ph:.0%} — {away_name} {pa:.0%} — Draw {pd_:.0%}.** "
        f"Most likely score **{top_score}** ({top_p:.0%})."
    )

    if supporting:
        lead = f"{fav} are favored" if (fav_p >= 0.5 or margin >= 0.08) else f"{fav} edge it"
        body = f"{lead}, driven mainly by {'; '.join(supporting[:3])}."
        if opposing:
            body += f" The gap isn't bigger because of {opposing[0]}."
    elif opposing:
        # Favorite leads on the blend but every named factor points the other way.
        body = f"It's close — {fav} lead narrowly despite {opposing[0]}."
    else:
        body = f"The teams look closely matched — {conf}."

    # Form line (always shown — it's display material).
    form_line = ""
    if fh.n and fa.n:
        form_line = (
            f" Recent form: {home_name} {fh.record}, {away_name} {fa.record}."
        )

    cold_note = ""
    if pred.cold_start:
        cold_note = (
            " ⚠️ One side has little recent data, so this leans on the Elo prior "
            "and carries extra uncertainty."
        )

    return f"{headline} {body}{form_line}{cold_note}"
