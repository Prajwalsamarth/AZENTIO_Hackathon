"""Unsupervised anomaly detection: Isolation Forest + autoencoder.

Both produce scores on arbitrary, incomparable scales, so each is converted to
a percentile rank against the distribution captured at fit time. That reference
distribution is persisted -- without it, a single transaction scored in
isolation has nothing to be a percentile *of*.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor

import config
import log

LOG = log.get("anomaly")


def _percentile_of(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Fraction of the reference distribution at or below each value."""
    if reference.size == 0:
        return np.zeros_like(values, dtype="float64")
    idx = np.searchsorted(reference, values, side="right")
    return idx / float(reference.size)


class AnomalyEnsemble:
    """Isolation Forest + a bottleneck autoencoder, fitted on the corpus.

    The two models see deliberately different feature spaces. The forest gets
    everything, including the 60 one-hot columns, because axis-aligned splits
    handle sparse indicators well. The autoencoder gets only the continuous and
    derived features: a standardized one-hot column that is 1 for 2% of rows is
    close to unpredictable, so an 8-unit bottleneck can only emit the mean and
    bank ~1.0 error on it every time. Including them buries the continuous
    deviation that actually carries signal under a constant noise floor.

    Measured effect of the split: the known fraud case moves from rank #69 to
    #12 of 988, and rank correlation with the forest falls 0.60 -> 0.48, so the
    ensemble gains real diversity rather than two views of the same thing.
    """

    def __init__(self, iforest=None, autoencoder=None, autoencoder_mask=None,
                 iforest_reference=None, autoencoder_reference=None):
        self.iforest = iforest
        self.autoencoder = autoencoder
        self.autoencoder_mask = autoencoder_mask
        self.iforest_reference = iforest_reference
        self.autoencoder_reference = autoencoder_reference

    # --- fit --------------------------------------------------------------
    def fit(self, X_scaled: np.ndarray, autoencoder_mask: np.ndarray = None):
        LOG.info(f"        isolation forest: {config.IFOREST_PARAMS['n_estimators']} trees "
                 f"on all {X_scaled.shape[1]} features")
        self.iforest = IsolationForest(**config.IFOREST_PARAMS).fit(X_scaled)

        self.autoencoder_mask = (np.ones(X_scaled.shape[1], dtype=bool)
                                 if autoencoder_mask is None else np.asarray(autoencoder_mask))
        X_ae = X_scaled[:, self.autoencoder_mask]
        LOG.info(f"        autoencoder: {config.AUTOENCODER_PARAMS['hidden_layer_sizes']} "
                 f"on {X_ae.shape[1]} continuous features "
                 f"(up to {config.AUTOENCODER_PARAMS['max_iter']} iterations)")
        self.autoencoder = MLPRegressor(**config.AUTOENCODER_PARAMS).fit(X_ae, X_ae)
        LOG.debug(f"autoencoder converged after {self.autoencoder.n_iter_} iterations, "
                  f"final loss {self.autoencoder.loss_:.5f}")

        self.iforest_reference = np.sort(self._iforest_raw(X_scaled))
        self.autoencoder_reference = np.sort(self._reconstruction_error(X_scaled))
        return self

    # --- raw scores -------------------------------------------------------
    def _iforest_raw(self, X: np.ndarray) -> np.ndarray:
        # score_samples is higher for normal points; negate so higher = stranger.
        return -self.iforest.score_samples(X)

    def _reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        return np.mean(self.per_feature_error(X), axis=1)

    def per_feature_error(self, X: np.ndarray) -> np.ndarray:
        """Squared error per autoencoder feature -- names which fields were odd."""
        X_ae = X[:, self.autoencoder_mask]
        return (self.autoencoder.predict(X_ae) - X_ae) ** 2

    # --- scoring ----------------------------------------------------------
    def score(self, X: np.ndarray) -> dict:
        iforest_raw = self._iforest_raw(X)
        autoencoder_raw = self._reconstruction_error(X)
        return {
            "iforest_raw": iforest_raw,
            "autoencoder_raw": autoencoder_raw,
            "iforest": _percentile_of(self.iforest_reference, iforest_raw),
            "autoencoder": _percentile_of(self.autoencoder_reference, autoencoder_raw),
        }


def fuse(rule: np.ndarray, iforest: np.ndarray, autoencoder: np.ndarray) -> np.ndarray:
    w = config.FUSION_WEIGHTS
    return (w["rule"] * np.asarray(rule, dtype="float64")
            + w["iforest"] * np.asarray(iforest, dtype="float64")
            + w["autoencoder"] * np.asarray(autoencoder, dtype="float64"))
