"""Dixon-Coles Poisson model — the scoreline engine (plan §6b).

Each team gets an attack and a defense parameter. For home team i vs away j:

    log λ (home goals) = attack_i - defense_j + γ      (γ = home adv, 0 on neutral)
    log μ (away goals) = attack_j - defense_i

Goals are Poisson(λ) / Poisson(μ), with the Dixon-Coles low-score correction τ
that fixes the well-known under-/over-estimation of 0-0, 1-0, 0-1, 1-1.

Two essentials for international football:
  - Time decay: each historical match is weighted exp(-ξ·age) so last month
    matters far more than four years ago.
  - Ridge regularization: pins down the model's additive degeneracy AND shrinks
    thin-data teams toward average (a clean cold-start guard, plan §2/§12).

Fit by maximizing the time-weighted log-likelihood with an analytic gradient
(fast + stable, so the backtest can refit cheaply).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date

import numpy as np
from scipy.optimize import minimize

from src.db.models import get_connection

# Defaults — tunable on the backtest (plan §8).
DEFAULT_XI = 0.0019      # time-decay per DAY (~half-life 1 year)
DEFAULT_REG = 0.01       # ridge strength
DEFAULT_MAX_YEARS = 12   # ignore matches older than this (decay makes them ~0 anyway)


@dataclass
class DixonColesModel:
    attack: dict[int, float]
    defense: dict[int, float]
    gamma: float
    rho: float
    fitted_teams: set[int] = field(default_factory=set)

    def expected_goals(
        self, home_id: int, away_id: int, neutral: bool
    ) -> tuple[float, float]:
        g = 0.0 if neutral else self.gamma
        # Missing teams (no recent matches) default to 0.0 = league-average — a
        # clean cold-start fallback. The predictor flags this and leans on Elo.
        ah = self.attack.get(home_id, 0.0); aa = self.attack.get(away_id, 0.0)
        dh = self.defense.get(home_id, 0.0); da = self.defense.get(away_id, 0.0)
        log_lam = ah - da + g
        log_mu = aa - dh
        return float(np.exp(log_lam)), float(np.exp(log_mu))

    def score_matrix(
        self, home_id: int, away_id: int, neutral: bool, max_goals: int = 10
    ) -> np.ndarray:
        """P(home=x, away=y) for x,y in 0..max_goals, with the DC correction."""
        lam, mu = self.expected_goals(home_id, away_id, neutral)
        return score_matrix_from_lambdas(lam, mu, self.rho, max_goals)


def _poisson_pmf_vec(lmbda: float, n: int) -> np.ndarray:
    """PMF for k=0..n via stable recurrence."""
    out = np.empty(n + 1)
    out[0] = np.exp(-lmbda)
    for k in range(1, n + 1):
        out[k] = out[k - 1] * lmbda / k
    return out


def score_matrix_from_lambdas(
    lam: float, mu: float, rho: float, max_goals: int = 10
) -> np.ndarray:
    home = _poisson_pmf_vec(lam, max_goals)
    away = _poisson_pmf_vec(mu, max_goals)
    mat = np.outer(home, away)
    # Dixon-Coles correction on the four low scores.
    mat[0, 0] *= 1.0 - lam * mu * rho
    mat[0, 1] *= 1.0 + lam * rho
    mat[1, 0] *= 1.0 + mu * rho
    mat[1, 1] *= 1.0 - rho
    mat = np.clip(mat, 1e-12, None)
    return mat / mat.sum()


# ---- Fitting --------------------------------------------------------------

def _load_training_arrays(conn, as_of: str | None, max_years: int):
    """Pull played matches up to `as_of` (exclusive of future — no leakage)."""
    sql = """SELECT date, home_id, away_id, home_goals, away_goals, neutral
             FROM matches
             WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL"""
    params: list = []
    if as_of:
        sql += " AND date < ?"
        params.append(as_of)
    rows = conn.execute(sql, params).fetchall()

    ref = date.fromisoformat(as_of) if as_of else date.today()
    cutoff_days = max_years * 365
    home, away, hg, ag, neu, age = [], [], [], [], [], []
    for r in rows:
        d = date.fromisoformat(r["date"])
        age_days = (ref - d).days
        if age_days > cutoff_days or age_days < 0:
            continue
        home.append(r["home_id"]); away.append(r["away_id"])
        hg.append(r["home_goals"]); ag.append(r["away_goals"])
        neu.append(1.0 if r["neutral"] else 0.0); age.append(age_days)
    return (np.array(home), np.array(away), np.array(hg, float),
            np.array(ag, float), np.array(neu), np.array(age, float))


def fit_dixon_coles(
    conn: sqlite3.Connection | None = None,
    as_of: str | None = None,
    xi: float = DEFAULT_XI,
    reg: float = DEFAULT_REG,
    max_years: int = DEFAULT_MAX_YEARS,
) -> DixonColesModel:
    """Fit attack/defense/γ/ρ on time-weighted history strictly before `as_of`."""
    own = conn is None
    if own:
        conn = get_connection()

    home, away, hg, ag, neu, age = _load_training_arrays(conn, as_of, max_years)
    if own:
        conn.close()
    if len(home) == 0:
        raise ValueError("No training matches in window.")

    # Map team ids -> contiguous indices.
    teams = sorted(set(home.tolist()) | set(away.tolist()))
    idx = {t: i for i, t in enumerate(teams)}
    T = len(teams)
    hi = np.array([idx[t] for t in home])
    ai = np.array([idx[t] for t in away])
    w = np.exp(-xi * age)  # time-decay weights

    # Param layout: [attack(T), defense(T), gamma, rho]
    def unpack(p):
        return p[:T], p[T:2 * T], p[2 * T], p[2 * T + 1]

    # Precompute which matches are the four DC-corrected scorelines.
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)

    def objective(p):
        att, dfn, gamma, rho = unpack(p)
        log_lam = att[hi] - dfn[ai] + gamma * (1.0 - neu)
        log_mu = att[ai] - dfn[hi]
        lam = np.exp(log_lam); mu = np.exp(log_mu)

        # Poisson log-likelihood (drop constant factorials).
        ll = hg * log_lam - lam + ag * log_mu - mu

        # DC correction term log τ.
        tau = np.ones_like(lam)
        tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
        tau[m01] = 1.0 + lam[m01] * rho
        tau[m10] = 1.0 + mu[m10] * rho
        tau[m11] = 1.0 - rho
        tau = np.clip(tau, 1e-10, None)
        ll = ll + np.log(tau)

        nll = -np.sum(w * ll) + 0.5 * reg * (np.sum(att ** 2) + np.sum(dfn ** 2))

        # ---- analytic gradient ----
        # base Poisson parts
        d_loglam = w * (hg - lam)      # ∂ll/∂logλ per match
        d_logmu = w * (ag - mu)
        # τ contributions (chain through λ, μ, ρ)
        dtau_dlam = np.zeros_like(lam); dtau_dmu = np.zeros_like(lam)
        dtau_drho = np.zeros_like(lam)
        inv = 1.0 / tau
        dtau_dlam[m00] = -mu[m00] * rho * inv[m00]
        dtau_dmu[m00] = -lam[m00] * rho * inv[m00]
        dtau_drho[m00] = -lam[m00] * mu[m00] * inv[m00]
        dtau_dlam[m01] = rho * inv[m01]
        dtau_drho[m01] = lam[m01] * inv[m01]
        dtau_dmu[m10] = rho * inv[m10]
        dtau_drho[m10] = mu[m10] * inv[m10]
        dtau_drho[m11] = -1.0 * inv[m11]
        # chain λ=exp(logλ): ∂logτ/∂logλ = dtau_dlam * λ
        d_loglam = d_loglam + w * dtau_dlam * lam
        d_logmu = d_logmu + w * dtau_dmu * mu

        g_att = np.zeros(T); g_dfn = np.zeros(T)
        # logλ = att[hi] - dfn[ai] + γ ; logμ = att[ai] - dfn[hi]
        np.add.at(g_att, hi, d_loglam)
        np.add.at(g_att, ai, d_logmu)
        np.add.at(g_dfn, ai, -d_loglam)
        np.add.at(g_dfn, hi, -d_logmu)
        g_gamma = np.sum(d_loglam * (1.0 - neu))
        g_rho = np.sum(w * dtau_drho)

        # negate (we minimize nll) and add ridge derivative
        grad = np.empty_like(p)
        grad[:T] = -g_att + reg * att
        grad[T:2 * T] = -g_dfn + reg * dfn
        grad[2 * T] = -g_gamma
        grad[2 * T + 1] = -g_rho
        return nll, grad

    x0 = np.zeros(2 * T + 2)
    x0[2 * T] = 0.25   # gamma start
    x0[2 * T + 1] = -0.05  # rho start
    # Keep rho in a safe range so τ stays positive.
    bounds = [(None, None)] * (2 * T) + [(None, None), (-0.2, 0.2)]
    res = minimize(objective, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 300})

    att, dfn, gamma, rho = unpack(res.x)
    return DixonColesModel(
        attack={t: float(att[idx[t]]) for t in teams},
        defense={t: float(dfn[idx[t]]) for t in teams},
        gamma=float(gamma),
        rho=float(rho),
        fitted_teams=set(teams),
    )


if __name__ == "__main__":
    conn = get_connection()
    print("Fitting Dixon-Coles on time-weighted history (this takes a few seconds)...")
    model = fit_dixon_coles(conn)
    names = {r["team_id"]: r["name"] for r in conn.execute("SELECT team_id, name FROM teams")}

    # Sanity: a marquee matchup on neutral ground.
    def show(a, b):
        ia = next(t for t, n in names.items() if n == a)
        ib = next(t for t, n in names.items() if n == b)
        if ia not in model.fitted_teams or ib not in model.fitted_teams:
            print(f"  ({a} or {b} not in recent window)"); return
        mat = model.score_matrix(ia, ib, neutral=True)
        ph = np.tril(mat, -1).sum(); pd_ = np.trace(mat); pa = np.triu(mat, 1).sum()
        lam, mu = model.expected_goals(ia, ib, neutral=True)
        top = np.unravel_index(mat.argmax(), mat.shape)
        print(f"  {a} vs {b}  (neutral)")
        print(f"    win {ph:.0%} / draw {pd_:.0%} / loss {pa:.0%}"
              f" | xG {lam:.2f}-{mu:.2f} | most likely {top[0]}-{top[1]} ({mat[top]:.0%})")

    print(f"\nFitted {len(model.fitted_teams)} teams. gamma(home adv)={model.gamma:.3f}, rho={model.rho:.3f}\n")
    show("Brazil", "Croatia")
    show("Spain", "Argentina")
    show("Germany", "Japan")
    conn.close()
