"""Estimators for the average treatment effect (ATE) of a binary treatment.

All estimators take a feature matrix X, a binary treatment vector A and a
binary outcome vector Y, and return a dict with the estimate, standard error
and 95% confidence interval.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

Z = 1.959964


def _result(est: float, se: float) -> dict:
    return {"estimate": est, "se": se, "lower": est - Z * se, "upper": est + Z * se}


# ---------------------------------------------------------------- data
def load_hillstrom(path: str = "../data/hillstrom.csv") -> pd.DataFrame:
    """Load the Hillstrom MineThatData e-mail dataset.

    If the CSV is missing, download it with scikit-uplift and cache it.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        from sklift.datasets import fetch_hillstrom

        b = fetch_hillstrom()
        df = pd.concat([b.data, b.target.rename("visit")], axis=1)
        df["segment"] = b.treatment
        df.to_csv(path, index=False)
    df = df.copy()
    df["treated"] = (df["segment"] != "No E-Mail").astype(int)
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Customer features known before the campaign was sent."""
    X = pd.DataFrame(
        {
            "recency": df["recency"],
            "log_history": np.log(df["history"]),
            "mens": df["mens"],
            "womens": df["womens"],
            "newbie": df["newbie"],
        }
    )
    X = X.join(pd.get_dummies(df["zip_code"], prefix="zip", drop_first=True, dtype=int))
    X = X.join(pd.get_dummies(df["channel"], prefix="channel", drop_first=True, dtype=int))
    return X


def simulate_targeting(df: pd.DataFrame, strength: float = 1.0, seed: int = 0) -> pd.DataFrame:
    """Turn the randomised experiment into a realistic *observational* dataset.

    Mimics a marketing team that preferentially e-mailed recent, high-value,
    multichannel customers. Each customer gets a targeting score p(x); e-mailed
    customers are kept with probability p(x)/2 and un-emailed customers with
    probability 1 - p(x). Because two thirds of customers were e-mailed in the
    experiment, every customer is kept with the same overall probability (1/3),
    so the subsample is a random third of the customer base, but within it the
    chance of having been e-mailed is exactly p(x). The true average effect is
    therefore unchanged and the full experiment still gives the right answer.
    """
    rng = np.random.default_rng(seed)
    lh = np.log(df["history"])
    score = (
        strength * (lh - lh.mean()) / lh.std()
        - strength * (df["recency"] - df["recency"].mean()) / df["recency"].std()
        + 0.5 * strength * (df["channel"] == "Multichannel")
    )
    p = expit(score)
    keep_prob = np.where(df["treated"] == 1, p / 2, 1 - p)
    return df[rng.uniform(size=len(df)) < keep_prob].reset_index(drop=True)


# ---------------------------------------------------------------- estimators
def naive(A, Y) -> dict:
    """Difference in outcome rates between e-mailed and not e-mailed customers."""
    A, Y = np.asarray(A), np.asarray(Y)
    p1, p0 = Y[A == 1].mean(), Y[A == 0].mean()
    se = np.sqrt(p1 * (1 - p1) / (A == 1).sum() + p0 * (1 - p0) / (A == 0).sum())
    return _result(p1 - p0, se)


def _default_learner():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=0)


def cross_fit_nuisance(X, A, Y, learner=_default_learner, n_folds: int = 5, seed: int = 0):
    """Out-of-fold predictions of the outcome model Q(a, x) and propensity g(x)."""
    X, A, Y = np.asarray(X, float), np.asarray(A), np.asarray(Y)
    n = len(Y)
    Q1, Q0, g = np.empty(n), np.empty(n), np.empty(n)
    for train, test in KFold(n_folds, shuffle=True, random_state=seed).split(X):
        q = learner().fit(np.column_stack([A[train], X[train]]), Y[train])
        Q1[test] = q.predict_proba(np.column_stack([np.ones(len(test)), X[test]]))[:, 1]
        Q0[test] = q.predict_proba(np.column_stack([np.zeros(len(test)), X[test]]))[:, 1]
        g[test] = learner().fit(X[train], A[train]).predict_proba(X[test])[:, 1]
    return Q1, Q0, np.clip(g, 0.01, 0.99)


def g_computation(Q1, Q0) -> dict:
    """Outcome-regression (plug-in) estimate. No valid SE, so bootstrap if needed."""
    return _result(float(np.mean(Q1 - Q0)), np.nan)


def ipw(A, Y, g) -> dict:
    """Normalised (Hajek) inverse probability weighting."""
    A, Y = np.asarray(A), np.asarray(Y)
    w1, w0 = A / g, (1 - A) / (1 - g)
    mu1, mu0 = np.sum(w1 * Y) / np.sum(w1), np.sum(w0 * Y) / np.sum(w0)
    ic = w1 * (Y - mu1) / np.mean(w1) - w0 * (Y - mu0) / np.mean(w0)
    return _result(mu1 - mu0, ic.std(ddof=1) / np.sqrt(len(Y)))


def aipw(A, Y, Q1, Q0, g) -> dict:
    """Augmented IPW (doubly robust, one-step estimator)."""
    A, Y = np.asarray(A), np.asarray(Y)
    QA = np.where(A == 1, Q1, Q0)
    phi = Q1 - Q0 + (A / g - (1 - A) / (1 - g)) * (Y - QA)
    est = phi.mean()
    return _result(est, phi.std(ddof=1) / np.sqrt(len(Y)))


def tmle(A, Y, Q1, Q0, g) -> dict:
    """Targeted maximum likelihood estimation for a binary outcome.

    Fluctuates the initial outcome model along the 'clever covariate' so the
    final plug-in estimate solves the efficient influence curve equation.
    """
    A, Y = np.asarray(A), np.asarray(Y, float)
    eps_clip = 1e-6
    Q1c, Q0c = np.clip(Q1, eps_clip, 1 - eps_clip), np.clip(Q0, eps_clip, 1 - eps_clip)
    QA = np.where(A == 1, Q1c, Q0c)
    H = A / g - (1 - A) / (1 - g)
    # one-parameter logistic fluctuation with offset logit(QA)
    offset = logit(QA)
    eps = 0.0
    for _ in range(50):  # Newton steps; converges in a few iterations
        p = expit(offset + eps * H)
        grad = np.sum(H * (Y - p))
        hess = np.sum(H**2 * p * (1 - p))
        step = grad / hess
        eps += step
        if abs(step) < 1e-10:
            break
    Q1s = expit(logit(Q1c) + eps / g)
    Q0s = expit(logit(Q0c) - eps / (1 - g))
    QAs = np.where(A == 1, Q1s, Q0s)
    est = float(np.mean(Q1s - Q0s))
    ic = H * (Y - QAs) + Q1s - Q0s - est
    return _result(est, ic.std(ddof=1) / np.sqrt(len(Y)))


def estimate_all(df: pd.DataFrame, outcome: str = "visit", learner=_default_learner, seed: int = 0) -> pd.DataFrame:
    """Run every estimator on one dataset and return a tidy table."""
    X, A, Y = make_features(df), df["treated"].to_numpy(), df[outcome].to_numpy()
    Q1, Q0, g = cross_fit_nuisance(X, A, Y, learner=learner, seed=seed)
    rows = {
        "Naive comparison": naive(A, Y),
        "Outcome regression": g_computation(Q1, Q0),
        "IPW": ipw(A, Y, g),
        "AIPW": aipw(A, Y, Q1, Q0, g),
        "TMLE": tmle(A, Y, Q1, Q0, g),
    }
    return pd.DataFrame(rows).T
