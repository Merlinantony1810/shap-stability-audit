"""
Model training.

One function, deliberately. Every model in the audit — the baseline and
every reformatted variant — goes through this same function with the same
hyperparameters, so that the encoding of the protected attribute is the
only thing that differs between comparisons.
"""

from sklearn.ensemble import RandomForestClassifier

from src import config


def train_random_forest(X_train, y_train, random_state=None, **rf_kwargs):
    """
    Fit a Random Forest on the given training data.

    Args:
        X_train, y_train: training features and target
        random_state:     model seed; defaults to config.RANDOM_SEED.
                          Exposed as an argument so the seed-stability
                          experiment can vary it without touching config.
        **rf_kwargs:      any other RandomForestClassifier setting, e.g.
                          max_depth or n_estimators. Needed because larger
                          datasets have to be capped to keep exact SHAP
                          tractable.

    Returns:
        A fitted RandomForestClassifier.
    """
    if random_state is None:
        random_state = config.RANDOM_SEED

    params = {
        "n_estimators": 300,
        "random_state": random_state,
        "n_jobs": -1,
    }
    params.update(rf_kwargs)

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model