"""
Benchmark4: Unified Classifier Wrapper.

6 classifiers with sklearn-compatible fit/predict/predict_proba interface:
  1. TabPFN     - Foundation model (Nature 2025). PCA-bagging for >2000D, ECOC for >10 classes.
  2. XGBoost    - GBDT with GPU support.
  3. CatBoost   - GBDT on CPU (#1 on TabArena).
  4. RandomForest - Classical ensemble baseline.
  5. TabM       - Ensembled MLP (ICLR 2025), via pytabkit.
  6. RealMLP    - Meta-tuned MLP (NeurIPS 2024), via pytabkit.

Usage:
    clf = get_classifier('TabPFN', n_features=200, n_classes=8, seed=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from typing import Dict, Optional, Any
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from RuleBenchmark.benchmark4.config import CLASSIFIERS


# =============================================================================
# Feature clipper (prevents float32 overflow after StandardScaler)
# =============================================================================

class FeatureClipper(BaseEstimator):
    """Clip features to prevent float32 overflow after StandardScaler.

    When StandardScaler encounters near-zero-variance columns in train,
    test samples with different values in those columns can produce
    astronomically large values that overflow float32.
    """
    def __init__(self, max_abs=1e6):
        self.max_abs = max_abs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.clip(X, -self.max_abs, self.max_abs)


# =============================================================================
# Library availability
# =============================================================================

def _check_lib(name):
    """Check if a library is installed using find_spec (no code execution).

    This avoids issues where full import triggers CUDA init that fails
    on batch nodes before GPU is properly allocated.
    """
    import importlib.util
    pkg_map = {
        'tabpfn': 'tabpfn',
        'xgboost': 'xgboost',
        'catboost': 'catboost',
        'pytabkit': 'pytabkit',
    }
    return importlib.util.find_spec(pkg_map.get(name, name)) is not None

HAS_TABPFN = _check_lib('tabpfn')
HAS_XGBOOST = _check_lib('xgboost')
HAS_CATBOOST = _check_lib('catboost')
HAS_PYTABKIT = _check_lib('pytabkit')


# =============================================================================
# TabPFN with PCA-Bagging
# =============================================================================

class TabPFNPCABagging(BaseEstimator, ClassifierMixin):
    """TabPFN with PCA-bagging for >2000D features.

    If n_features <= 2000: use TabPFN directly.
    If n_features > 2000: project to 500D via random PCA, run TabPFN
        multiple times, average predictions.
    """

    def __init__(self, n_projections=5, projection_dim=500, seed=42,
                 n_estimators=32):
        self.n_projections = n_projections
        self.projection_dim = projection_dim
        self.seed = seed
        self.n_estimators = n_estimators
        self._needs_bagging = False
        self._projections = []
        self._models = []
        self._single_model = None

    def fit(self, X, y):
        from tabpfn import TabPFNClassifier
        self.classes_ = np.unique(y)
        n_features = X.shape[1]

        if n_features <= 2000:
            self._needs_bagging = False
            self._single_model = TabPFNClassifier(
                device='cuda', n_estimators=self.n_estimators)
            self._single_model.fit(X, y)
        else:
            self._needs_bagging = True
            rng = np.random.RandomState(self.seed)
            self._projections = []
            self._models = []

            for i in range(self.n_projections):
                # Random projection via PCA-like sampling
                proj_dim = min(self.projection_dim, n_features)
                indices = rng.choice(n_features, proj_dim, replace=False)
                indices.sort()
                self._projections.append(indices)

                X_proj = X[:, indices]
                model = TabPFNClassifier(device='cuda', n_estimators=self.n_estimators)
                model.fit(X_proj, y)
                self._models.append(model)

        return self

    def predict_proba(self, X):
        if not self._needs_bagging:
            return self._single_model.predict_proba(X)
        else:
            proba_sum = None
            for proj, model in zip(self._projections, self._models):
                X_proj = X[:, proj]
                proba = model.predict_proba(X_proj)
                if proba_sum is None:
                    proba_sum = proba
                else:
                    proba_sum += proba
            return proba_sum / len(self._models)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# =============================================================================
# Classifier Factory
# =============================================================================

def get_classifier(
    name: str,
    n_features: int = 100,
    n_classes: int = 2,
    seed: int = 42,
    device: str = 'cuda',
) -> Pipeline:
    """Create a classifier pipeline with preprocessing.

    Returns a sklearn Pipeline with:
        1. SimpleImputer(strategy='median') — handles NaN
        2. StandardScaler — normalizes features
        3. Classifier

    Args:
        name: Classifier name (TabPFN, CatBoost, RandomForest, TabM, ModernNCA)
        n_features: Number of input features (used for TabPFN limit check)
        n_classes: Number of classes (used for TabPFN ECOC check)
        seed: Random seed
        device: 'cuda' or 'cpu'
    """
    clf = _build_classifier(name, n_features, n_classes, seed, device)

    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clipper', FeatureClipper(max_abs=1e6)),
        ('classifier', clf),
    ])


def _build_classifier(name, n_features, n_classes, seed, device):
    """Build the raw classifier (no preprocessing)."""

    if name == 'TabPFN':
        if not HAS_TABPFN:
            raise ImportError("TabPFN not installed: pip install tabpfn")
        if n_classes > 10:
            # ECOC mode: reduced n_estimators=8 for speed (benchmark3 pattern)
            base = TabPFNPCABagging(
                n_projections=CLASSIFIERS['TabPFN']['pca_bagging']['n_projections'],
                projection_dim=CLASSIFIERS['TabPFN']['pca_bagging']['projection_dim'],
                seed=seed,
                n_estimators=8,
            )
            from sklearn.multiclass import OutputCodeClassifier
            return OutputCodeClassifier(base, code_size=1.5, random_state=seed, n_jobs=-1)
        else:
            return TabPFNPCABagging(
                n_projections=CLASSIFIERS['TabPFN']['pca_bagging']['n_projections'],
                projection_dim=CLASSIFIERS['TabPFN']['pca_bagging']['projection_dim'],
                seed=seed,
                n_estimators=CLASSIFIERS['TabPFN']['params'].get('n_estimators', 16),
            )

    elif name == 'XGBoost':
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed: pip install xgboost")
        from xgboost import XGBClassifier
        params = dict(CLASSIFIERS['XGBoost']['params'])
        params['random_state'] = seed
        # Force CPU always — GPU OOM on shared cluster nodes
        params['device'] = 'cpu'
        params['tree_method'] = 'hist'
        if n_classes <= 2:
            params['objective'] = 'binary:logistic'
            params['eval_metric'] = 'logloss'
        else:
            params['num_class'] = n_classes
        return XGBClassifier(**params)

    elif name == 'CatBoost':
        if not HAS_CATBOOST:
            raise ImportError("CatBoost not installed: pip install catboost")
        from catboost import CatBoostClassifier
        params = dict(CLASSIFIERS['CatBoost']['params'])
        params['random_seed'] = seed
        return CatBoostClassifier(**params)

    elif name == 'RandomForest':
        from sklearn.ensemble import RandomForestClassifier
        params = dict(CLASSIFIERS['RandomForest']['params'])
        params['random_state'] = seed
        return RandomForestClassifier(**params)

    elif name == 'TabM':
        if not HAS_PYTABKIT:
            raise ImportError("pytabkit not installed: pip install pytabkit")
        from pytabkit.models.sklearn.sklearn_interfaces import TabM_D_Classifier
        params = dict(CLASSIFIERS['TabM']['params'])
        return TabM_D_Classifier(random_state=seed, device=device, **params)

    elif name == 'RealMLP':
        if not HAS_PYTABKIT:
            raise ImportError("pytabkit not installed: pip install pytabkit")
        from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Classifier
        params = dict(CLASSIFIERS['RealMLP']['params'])
        return RealMLP_TD_Classifier(random_state=seed, device=device, **params)

    else:
        raise ValueError(f"Unknown classifier: {name}. "
                         f"Available: {list(CLASSIFIERS.keys())}")


def get_available_classifiers() -> list:
    """Get list of classifiers that are currently importable."""
    available = ['RandomForest']  # Always available via sklearn
    if HAS_TABPFN:
        available.append('TabPFN')
    if HAS_XGBOOST:
        available.append('XGBoost')
    if HAS_CATBOOST:
        available.append('CatBoost')
    if HAS_PYTABKIT:
        available.append('TabM')
        available.append('RealMLP')
    return available


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score

    print("=" * 60)
    print("  Classifier Wrapper Test")
    print("=" * 60)

    available = get_available_classifiers()
    print(f"  Available: {available}")
    missing = set(CLASSIFIERS.keys()) - set(available)
    if missing:
        print(f"  Missing: {missing}")

    # Small test dataset
    X, y = make_classification(
        n_samples=200, n_features=50, n_informative=20,
        n_classes=5, random_state=42)

    for name in available:
        try:
            clf = get_classifier(name, n_features=50, n_classes=5, seed=42,
                                 device='cpu')
            scores = cross_val_score(clf, X, y, cv=3, scoring='balanced_accuracy')
            print(f"  {name:15s}  bal_acc={scores.mean():.3f} +/- {scores.std():.3f}")
        except Exception as e:
            print(f"  {name:15s}  ERROR: {e}")

    # Test PCA-bagging with high-dim features
    if HAS_TABPFN:
        print("\n  Testing TabPFN PCA-bagging (3000D features)...")
        X_hd, y_hd = make_classification(
            n_samples=200, n_features=3000, n_informative=50,
            n_classes=5, random_state=42)
        try:
            clf = get_classifier('TabPFN', n_features=3000, n_classes=5,
                                 seed=42, device='cpu')
            scores = cross_val_score(clf, X_hd, y_hd, cv=3,
                                     scoring='balanced_accuracy')
            print(f"    PCA-bagging bal_acc={scores.mean():.3f}")
        except Exception as e:
            print(f"    PCA-bagging ERROR: {e}")

    print("\n  Classifier wrapper test DONE")
