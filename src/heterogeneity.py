"""Who responds most? Customer-level email effects from a causal forest.

The causal forest is learned from a *targeted* campaign (observational data),
then checked against customers it never saw, who come from the randomised
experiment and so give an unbiased answer within any group we choose.

Treatment has three arms: no email (0), men's email (1) and women's email (2).
Within the targeted campaign the choice between the two emails is still random,
but whether a customer was emailed at all depends on their value and recency,
so the forest has to adjust for that confounding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from causal import make_features, naive, simulate_targeting

ARMS = {"No E-Mail": 0, "Mens E-Mail": 1, "Womens E-Mail": 2}
EMAILS = {"Men's email": "Mens E-Mail", "Women's email": "Womens E-Mail"}


def split_customers(df: pd.DataFrame, frac: float = 0.5, seed: int = 0):
    """Randomly split customers into a training half (to be targeted) and a held-out half."""
    m = np.random.default_rng(seed).uniform(size=len(df)) < frac
    return df[m].reset_index(drop=True), df[~m].reset_index(drop=True)


def fit_causal_forest(obs: pd.DataFrame, outcome: str = "visit", min_samples_leaf: int = 300,
                      n_estimators: int = 1000, seed: int = 0) -> CausalForestDML:
    """Causal forest (EconML's CausalForestDML) with gradient-boosted nuisance models.

    The outcome and treatment models are cross-fitted and their residuals are
    used to grow honest trees that split on differences in the email effect.
    A large `min_samples_leaf` matters: with smaller leaves the forest chases
    noise and badly overstates how much the effect varies between customers.
    """
    learner = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=seed)
    cf = CausalForestDML(
        model_y=HistGradientBoostingRegressor(**learner),
        model_t=HistGradientBoostingClassifier(**learner),
        discrete_treatment=True, categories=[0, 1, 2], cv=5,
        n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, random_state=seed,
    )
    cf.fit(obs[outcome].to_numpy(), obs["segment"].map(ARMS).to_numpy(), X=make_features(obs).to_numpy(float))
    cf.feature_names_ = list(make_features(obs).columns)
    return cf


def predict_lifts(cf: CausalForestDML, df: pd.DataFrame) -> pd.DataFrame:
    """Predicted lift in visit probability from each email, for every customer in df."""
    X = make_features(df).reindex(columns=cf.feature_names_, fill_value=0).to_numpy(float)
    return pd.DataFrame({name: cf.effect(X, T0=0, T1=ARMS[seg]) for name, seg in EMAILS.items()}, index=df.index)


def lift_by_predicted_group(test: pd.DataFrame, lifts: pd.DataFrame, n_groups: int = 5,
                            outcome: str = "visit") -> pd.DataFrame:
    """Sort held-out customers into equal groups by predicted lift and compare the
    forest's prediction with the lift actually seen in the randomised experiment."""
    rows = []
    for name, seg in EMAILS.items():
        s = test.loc[test["segment"].isin([seg, "No E-Mail"])]
        pred = lifts.loc[s.index, name]
        group = pd.qcut(pred.rank(method="first"), n_groups, labels=False)
        for k, g in s.groupby(group):
            rows.append({"email": name, "group": k + 1, "customers": len(g), "predicted": pred[g.index].mean(),
                         **naive((g["segment"] == seg).astype(int), g[outcome])})
    return pd.DataFrame(rows)


def run_split(df: pd.DataFrame, split_seed: int, target_seed: int = 7, **forest_kw) -> pd.DataFrame:
    """Whole pipeline for one split: target the training half, fit, check on the held-out half."""
    train, test = split_customers(df, seed=split_seed)
    cf = fit_causal_forest(simulate_targeting(train, seed=target_seed), **forest_kw)
    return lift_by_predicted_group(test, predict_lifts(cf, test)).assign(split=split_seed)
