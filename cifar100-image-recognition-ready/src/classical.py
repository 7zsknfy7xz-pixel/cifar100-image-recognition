from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .features import make_flat_gray_features, make_hog_color_features


def _fit_model(
    model: Pipeline,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    tune: bool,
    grid: dict,
    cv: int = 3,
) -> Pipeline:
    if not tune:
        model.fit(train_features, train_labels)
        return model

    search = GridSearchCV(
        model,
        grid,
        cv=cv,
        error_score="raise",
        n_jobs=-1,
        scoring="accuracy",
        verbose=1,
    )
    search.fit(train_features, train_labels)
    print(f"Best classical params: {search.best_params_}; cv_acc={search.best_score_:.4f}")
    return search.best_estimator_


def train_pca_svm(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    tune: bool = False,
) -> np.ndarray:
    x_train = make_flat_gray_features(train_images)
    x_test = make_flat_gray_features(test_images)
    n_components = min(160, x_train.shape[0] - 1, x_train.shape[1])
    cv = 3
    min_cv_train_size = x_train.shape[0] - int(np.ceil(x_train.shape[0] / cv))
    tuned_component_cap = max(1, min(n_components, min_cv_train_size))
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=42)),
            ("svm", LinearSVC(C=1.0, max_iter=10000, random_state=42)),
        ]
    )
    component_grid = sorted({min(value, tuned_component_cap) for value in (80, 120, 160) if value > 0})
    grid = {"pca__n_components": component_grid, "svm__C": [0.3, 1.0, 3.0]}
    model = _fit_model(model, x_train, train_labels, tune, grid, cv=cv)
    return model.predict(x_test)


def train_hog_svm(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    tune: bool = False,
) -> np.ndarray:
    x_train = make_hog_color_features(train_images)
    x_test = make_hog_color_features(test_images)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", LinearSVC(C=1.0, max_iter=10000, random_state=42)),
        ]
    )
    model = _fit_model(model, x_train, train_labels, tune, {"svm__C": [0.3, 1.0, 3.0]})
    return model.predict(x_test)
