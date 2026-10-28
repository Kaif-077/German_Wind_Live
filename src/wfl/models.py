"""Model definitions: two simple baselines and a gradient-boosting model."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression


def make_gbm(loss="squared_error", quantile=None, max_leaf_nodes=15, max_iter=400,
             learning_rate=0.05, min_samples_leaf=40, l2=1.0, seed=0):
    kwargs = dict(
        loss=loss, max_leaf_nodes=max_leaf_nodes, max_iter=max_iter,
        learning_rate=learning_rate, min_samples_leaf=min_samples_leaf,
        l2_regularization=l2, early_stopping=False, random_state=seed,
    )
    if loss == "quantile":
        kwargs["quantile"] = quantile
    return HistGradientBoostingRegressor(**kwargs)


class PowerCurveBaseline:
    """Simple 'physics' baseline: capacity factor as a monotone function of the average
    forecast wind speed across locations (isotonic regression)."""

    def fit(self, ws_mean: pd.Series, capacity_factor: pd.Series):
        m = ws_mean.notna() & capacity_factor.notna()
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0)
        self.iso.fit(ws_mean[m].to_numpy(), capacity_factor[m].to_numpy())
        return self

    def predict(self, ws_mean: pd.Series) -> np.ndarray:
        return self.iso.predict(ws_mean.fillna(ws_mean.mean()).to_numpy())
