"""Version-specific model and detector factories; identical settings across datasets."""

from river import drift, forest, linear_model, multiclass, naive_bayes, neighbors, optim, preprocessing, tree

REGRESSION_MODELS = ("HoeffdingTree", "AdaptiveRandomForest", "LinearAdam", "KNN")
CLASSIFICATION_MODELS = ("HoeffdingTree", "AdaptiveRandomForest", "LogisticAdamOvR", "GaussianNB")
DETECTORS = {
    "regression": ("ADWIN", "PageHinkley", "KSWIN"),
    "classification": ("ADWIN", "DDM", "EDDM"),
}


def make_model(task, name, seed, config):
    if task not in DETECTORS:
        raise ValueError(f'Unknown task: {task}')
    allowed = REGRESSION_MODELS if task == 'regression' else CLASSIFICATION_MODELS
    if name not in allowed:
        raise ValueError(f'Unknown {task} model: {name}')
    common = dict(grace_period=100, max_depth=12, max_size=64)
    if name == "HoeffdingTree":
        if task == "regression":
            return tree.HoeffdingTreeRegressor(**common, leaf_prediction="mean")
        return tree.HoeffdingTreeClassifier(**common, leaf_prediction="nba")
    if name == "AdaptiveRandomForest":
        kwargs = dict(**common, n_models=config["forest_trees"], seed=seed,
                      drift_detector=drift.ADWIN(delta=0.002), warning_detector=drift.ADWIN(delta=0.01))
        if task == "regression":
            return forest.ARFRegressor(**kwargs, leaf_prediction="mean")
        return forest.ARFClassifier(**kwargs, leaf_prediction="nba")
    if name == "LinearAdam":
        return preprocessing.StandardScaler() | preprocessing.TargetStandardScaler(
            regressor=linear_model.LinearRegression(optimizer=optim.Adam(lr=0.03), intercept_lr=0.03)
        )
    if name == "KNN":
        return preprocessing.StandardScaler() | neighbors.KNNRegressor(
            n_neighbors=5, engine=neighbors.LazySearch(window_size=config["knn_window"])
        )
    if name == "LogisticAdamOvR":
        return preprocessing.StandardScaler() | multiclass.OneVsRestClassifier(
            classifier=linear_model.LogisticRegression(optimizer=optim.Adam(lr=0.01))
        )
    if name == "GaussianNB":
        return naive_bayes.GaussianNB()
    raise ValueError(name)


def make_detector(name, seed):
    if name == "ADWIN":
        return drift.ADWIN(delta=0.002)
    if name == "PageHinkley":
        return drift.PageHinkley(min_instances=30, delta=0.005, threshold=10, alpha=0.9999, mode="both")
    if name == "KSWIN":
        return drift.KSWIN(alpha=0.005, window_size=100, stat_size=30, seed=seed)
    if name == "DDM":
        return drift.binary.DDM(warm_start=30, warning_threshold=2, drift_threshold=3)
    if name == "EDDM":
        return drift.binary.EDDM(warm_start=30, alpha=0.95, beta=0.9)
    raise ValueError(name)
