"""Monte Carlo tournament simulator (plan §6f / §7).

Simulates the whole bracket many times from per-match probabilities to produce
'win group', 'reach SF/Final', 'win the tournament' odds.

The engine is format-agnostic: give it groups (name -> list of team_ids) and the
World Cup 2026 advancement rules (12 groups of 4; top 2 + 8 best third-placed go
through to a 32-team knockout). Per-match outcomes come from the same predictor
used everywhere else, so the tournament odds are a direct roll-up of the match
model — which makes them a strong sanity check on it.

Plugging in the real 2026 draw (via football-data standings or manual entry)
activates the headline odds; the engine itself is fully general and tested.
"""
from __future__ import annotations

import random
from collections import defaultdict

from src.model.predictor import ModelBundle, predict

STAGES = ["R32", "R16", "QF", "SF", "F", "W"]


class _Cache:
    """Memoize deterministic per-pairing probabilities so the Monte Carlo loop
    samples instead of re-running the model millions of times."""

    def __init__(self, bundle, conn):
        self.bundle = bundle
        self.conn = conn
        self.group: dict[tuple[int, int], tuple[float, float, float]] = {}
        self.ko: dict[tuple[int, int], float] = {}

    def group_probs(self, a, b):
        key = (a, b)
        if key not in self.group:
            p = predict(self.bundle, a, b, neutral=True, conn=self.conn)
            self.group[key] = (p.p_home, p.p_draw, p.p_away)
        return self.group[key]

    def ko_adv(self, a, b):
        key = (a, b)
        if key not in self.ko:
            p = predict(self.bundle, a, b, neutral=True, knockout=True, conn=self.conn)
            self.ko[key] = p.p_adv_home
        return self.ko[key]


def _sample_group_match(cache, a, b, rng):
    ph, pd_, pa = cache.group_probs(a, b)
    r = rng.random()
    if r < ph:
        return (3, 0)
    if r < ph + pd_:
        return (1, 1)
    return (0, 3)


def _sample_ko(cache, a, b, rng):
    return a if rng.random() < cache.ko_adv(a, b) else b


def _simulate_group(cache, bundle, teams, rng):
    """Round-robin; return team_ids ranked 1st..last (ties broken by Elo)."""
    pts = defaultdict(int)
    for t in teams:
        pts[t] = 0
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            hp, ap = _sample_group_match(cache, teams[i], teams[j], rng)
            pts[teams[i]] += hp
            pts[teams[j]] += ap
    # Rank by points, break ties by Elo (a reasonable proxy for GD/H2H here).
    return sorted(teams, key=lambda t: (pts[t], bundle.elo_of(t)), reverse=True)


def _seed_knockout(qualifiers, bundle):
    """Order qualifiers by strength and pair strongest vs weakest each round."""
    return sorted(qualifiers, key=lambda t: bundle.elo_of(t), reverse=True)


def simulate(
    bundle: ModelBundle,
    groups: dict[str, list[int]],
    n_sims: int = 10000,
    conn=None,
    seed: int = 0,
) -> dict[int, dict[str, float]]:
    """Run the tournament n_sims times. Returns per-team probabilities:
    {team_id: {'win_group':p, 'R32':p, 'R16':p, 'QF':p, 'SF':p, 'F':p, 'W':p}}."""
    own = conn is None
    if own:
        from src.db.models import get_connection
        conn = get_connection()
    rng = random.Random(seed)
    cache = _Cache(bundle, conn)

    counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for _ in range(n_sims):
        # ---- group stage ----
        thirds = []  # (points-ish rank proxy, team) for best-third selection
        qualifiers = []
        for name, teams in groups.items():
            ranked = _simulate_group(cache, bundle, teams, rng)
            counts[ranked[0]]["win_group"] += 1
            qualifiers.extend(ranked[:2])  # top 2 auto-qualify
            if len(ranked) >= 3:
                thirds.append(ranked[2])

        # 8 best third-placed teams (approximated by Elo — no GD tracked here).
        thirds_sorted = sorted(thirds, key=lambda t: bundle.elo_of(t), reverse=True)
        qualifiers.extend(thirds_sorted[:8])

        # Pad/truncate to a power of two for a clean single-elim bracket.
        bracket = _seed_knockout(qualifiers, bundle)
        size = 1
        while size * 2 <= len(bracket):
            size *= 2
        bracket = bracket[:size]

        # A team "reaches" the stage named by the size of the round it ENTERS.
        # (32→R32, 16→R16, 8→QF, 4→SF, 2→F, 1→champion). Correct for the real
        # 32-team knockout and for any smaller bracket alike.
        size_to_stage = {32: "R32", 16: "R16", 8: "QF", 4: "SF", 2: "F", 1: "W"}
        s = size_to_stage.get(len(bracket))
        if s:
            for t in bracket:
                counts[t][s] += 1

        # ---- knockout rounds ----
        round_teams = bracket
        while len(round_teams) > 1:
            nxt = []
            half = len(round_teams) // 2
            for k in range(half):
                a = round_teams[k]
                b = round_teams[len(round_teams) - 1 - k]  # strongest vs weakest
                nxt.append(_sample_ko(cache, a, b, rng))
            reached = size_to_stage.get(len(nxt))  # winners enter a round of size len(nxt)
            if reached:
                for t in nxt:
                    counts[t][reached] += 1
            round_teams = _seed_knockout(nxt, bundle)

    if own:
        conn.close()

    out: dict[int, dict[str, float]] = {}
    for t, c in counts.items():
        out[t] = {k: c.get(k, 0) / n_sims for k in ["win_group", *STAGES]}
    return out


def example_groups(conn, names: list[list[str]]) -> dict[str, list[int]]:
    """Build a groups dict from team names (helper for demos / manual entry)."""
    name_to_id = {r["name"]: r["team_id"] for r in conn.execute("SELECT team_id,name FROM teams")}
    groups = {}
    for i, grp in enumerate(names):
        groups[chr(ord("A") + i)] = [name_to_id[n] for n in grp]
    return groups


if __name__ == "__main__":
    import time
    from src.db.models import get_connection
    from src.model.predictor import build_models

    # Illustrative 8 groups of 4 (real teams; NOT the official 2026 draw).
    demo = [
        ["Brazil", "Switzerland", "Cameroon", "Serbia"],
        ["Argentina", "Mexico", "Poland", "Saudi Arabia"],
        ["France", "Denmark", "Tunisia", "Australia"],
        ["Spain", "Germany", "Japan", "Costa Rica"],
        ["England", "Netherlands", "Senegal", "Iran"],
        ["Portugal", "Uruguay", "Ghana", "South Korea"],
        ["Belgium", "Croatia", "Morocco", "Canada"],
        ["Colombia", "Ecuador", "United States", "Wales"],
    ]
    conn = get_connection()
    print("Building models + simulating 10,000 tournaments...")
    bundle = build_models(conn)
    groups = example_groups(conn, demo)
    t0 = time.time()
    res = simulate(bundle, groups, n_sims=10000, conn=conn)
    names = {r["team_id"]: r["name"] for r in conn.execute("SELECT team_id,name FROM teams")}
    conn.close()

    ranked = sorted(res.items(), key=lambda kv: kv[1]["W"], reverse=True)
    print(f"\n  Done in {time.time()-t0:.1f}s. Title odds (top 12):\n")
    print(f"  {'team':<16} {'win grp':>8} {'reach SF':>9} {'final':>7} {'WIN':>7}")
    print("  " + "-" * 50)
    for tid, p in ranked[:12]:
        print(f"  {names[tid]:<16} {p['win_group']:>7.0%} {p['SF']:>9.0%} "
              f"{p['F']:>7.0%} {p['W']:>6.1%}")

    # consistency check
    bad = [names[t] for t, p in res.items()
           if not (p["SF"] >= p["F"] - 1e-9 and p["F"] >= p["W"] - 1e-9)]
    print("\n  Round-consistency (SF>=F>=W) holds for all teams"
          if not bad else f"\n  ⚠️ inconsistency: {bad[:5]}")
