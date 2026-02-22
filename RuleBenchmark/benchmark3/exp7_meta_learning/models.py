"""Meta-learning models: GBR, RandomForest ensemble, DecisionTree."""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.preprocessing import StandardScaler

from .config import GBR_PARAMS, RF_PARAMS, TREE_PARAMS, N_RF_SEEDS, N_FEATURES


class MetaLearner:
    """Meta-learner that predicts descriptor performance from dataset features.

    Input: 40D vector = [25 dataset features | 15 descriptor one-hot]
    Output: predicted balanced_accuracy score

    Models:
        - GBR: Primary predictor (gradient boosting regressor)
        - RF ensemble: For uncertainty estimation (N_RF_SEEDS models)
        - DecisionTree: For interpretable rules
    """

    def __init__(self, gbr_params=None, rf_params=None, tree_params=None,
                 n_rf_seeds=N_RF_SEEDS):
        if gbr_params is None:
            gbr_params = GBR_PARAMS
        if rf_params is None:
            rf_params = RF_PARAMS
        if tree_params is None:
            tree_params = TREE_PARAMS

        self.gbr = GradientBoostingRegressor(**gbr_params)
        self.rf_ensemble = [
            RandomForestRegressor(**rf_params, random_state=s)
            for s in range(n_rf_seeds)
        ]
        self.tree = DecisionTreeRegressor(**tree_params)
        self.scaler = StandardScaler()
        self._fitted = False

    def _scale_features(self, X, fit=False):
        """Scale only the first 25 columns (continuous features), leave one-hot intact."""
        X_scaled = X.copy()
        if fit:
            self.scaler.fit(X[:, :N_FEATURES])
        X_scaled[:, :N_FEATURES] = self.scaler.transform(X[:, :N_FEATURES])
        return X_scaled

    def fit(self, X, y):
        """Fit all models. X: (n, 40), y: (n,)."""
        X_scaled = self._scale_features(X, fit=True)

        self.gbr.fit(X_scaled, y)
        for rf in self.rf_ensemble:
            rf.fit(X_scaled, y)
        self.tree.fit(X_scaled, y)

        self._fitted = True
        return self

    def predict(self, X):
        """GBR predictions. X: (n, 40)."""
        assert self._fitted, "Model not fitted"
        X_scaled = self._scale_features(X, fit=False)
        return self.gbr.predict(X_scaled)

    def predict_with_uncertainty(self, X):
        """RF ensemble predictions with uncertainty.

        Returns:
            mean_pred: (n,) mean prediction across ensemble
            std_pred: (n,) std across ensemble (uncertainty)
        """
        assert self._fitted, "Model not fitted"
        X_scaled = self._scale_features(X, fit=False)

        preds = np.array([rf.predict(X_scaled) for rf in self.rf_ensemble])
        mean_pred = preds.mean(axis=0)
        std_pred = preds.std(axis=0)
        return mean_pred, std_pred

    def predict_tree(self, X):
        """DecisionTree predictions."""
        assert self._fitted, "Model not fitted"
        X_scaled = self._scale_features(X, fit=False)
        return self.tree.predict(X_scaled)

    def get_feature_importance(self, feature_names=None):
        """Get feature importances from GBR.

        Args:
            feature_names: list of all 40 feature names (25 continuous + 15 one-hot)

        Returns:
            list of (name, importance) sorted by importance descending
        """
        assert self._fitted, "Model not fitted"
        importances = self.gbr.feature_importances_

        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(importances))]

        pairs = list(zip(feature_names, importances))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs

    def get_tree_rules(self, feature_names=None, max_depth=None):
        """Get human-readable decision tree rules.

        Args:
            feature_names: list of all 40 feature names
            max_depth: max depth to display (None = full tree)

        Returns:
            str: text representation of tree rules
        """
        assert self._fitted, "Model not fitted"
        # Use tree's actual depth if max_depth not specified
        # (some sklearn versions don't handle max_depth=None in export_text)
        if max_depth is None:
            max_depth = self.tree.get_depth()
        return export_text(
            self.tree,
            feature_names=feature_names,
            max_depth=max_depth,
        )

    def get_all_feature_names(self, descriptor_list):
        """Build the full 40-element feature name list.

        Args:
            descriptor_list: list of 15 descriptor names

        Returns:
            list of 40 names: FEATURE_NAMES + ["desc_" + d for d in descriptor_list]
        """
        from .config import FEATURE_NAMES
        return list(FEATURE_NAMES) + [f"desc_{d}" for d in descriptor_list]
