"""Explicit mappings and documented non-equivalences between River and CapyMOA."""

import math
from .runtime import ROOT
from .configuration import require, FAMILIES, AUTOML, CLUSTERS, LIBRARIES


def classifier(library, family, schema, dimension, seed, config):
    require(library, LIBRARIES, 'library')
    require(family, FAMILIES, 'classification family')
    if config.get('tree_max_depth') is not None:
        raise ValueError('Non-null tree_max_depth is unsupported in the paired comparison')
    common = dict(grace_period=config["tree_grace_period"], max_depth=None,
                  max_size=config["tree_memory_mib"], delta=1e-7, tau=0.05, leaf_prediction="nba")
    if library == "River":
        from river import tree, forest, linear_model, multiclass, naive_bayes, optim, drift
        if family == "HoeffdingTree":
            return tree.HoeffdingTreeClassifier(**common)
        if family == "AdaptiveRandomForest":
            common["delta"] = 0.01
            return forest.ARFClassifier(**common, n_models=config["forest_trees"],
                        max_features=max(1, int(math.sqrt(dimension))), lambda_value=6, seed=seed,
                        drift_detector=drift.ADWIN(delta=0.002), warning_detector=drift.ADWIN(delta=0.01))
        if family == "LinearLogistic":
            return multiclass.OneVsRestClassifier(linear_model.LogisticRegression(optimizer=optim.Adam(lr=0.01), l2=0))
        return naive_bayes.GaussianNB()
    from capymoa.classifier import HoeffdingTree, AdaptiveRandomForestClassifier, NaiveBayes, SGDClassifier
    from capymoa.base import MOAClassifier
    from capymoa.drift.detectors import ADWIN
    if family == "HoeffdingTree":
        return HoeffdingTree(schema=schema, random_seed=seed, grace_period=config["tree_grace_period"], confidence=1e-7,
                    tie_threshold=0.05, leaf_prediction="NaiveBayesAdaptive", max_byte_size=config["tree_memory_mib"] * 2**20,
                    stop_mem_management=False, disable_prepruning=False)
    if family == "AdaptiveRandomForest":
        from moa.classifiers.trees import ARFHoeffdingTree
        base = MOAClassifier(schema=schema, random_seed=seed, moa_learner=ARFHoeffdingTree,
               CLI=f"-g {config['tree_grace_period']} -c 0.01 -t 0.05 -l NBAdaptive -m {config['tree_memory_mib'] * 2**20}")
        return AdaptiveRandomForestClassifier(schema=schema, random_seed=seed, base_learner=base,
                   ensemble_size=config["forest_trees"], max_features=max(1, int(math.sqrt(dimension))),
                   lambda_param=6, number_of_jobs=1, drift_detection_method=ADWIN(delta=0.002),
                   warning_detection_method=ADWIN(delta=0.01))
    if family == "LinearLogistic":
        # Closest available adaptive-rate logistic model, not an Adam implementation.
        return SGDClassifier(schema=schema, loss="log_loss", penalty=None, learning_rate="invscaling",
                             eta0=0.01, n_jobs=1, random_seed=seed)
    return NaiveBayes(schema=schema, random_seed=seed)


def automl(name, schema, seed, n):
    require(name, AUTOML, 'AutoML model')
    from capymoa.classifier import HoeffdingTree, HoeffdingAdaptiveTree, NaiveBayes
    from capymoa.automl import EpsilonGreedy
    from .automl_adapters import RecordedBandit, ActiveSuccessiveHalving, SeededAutoClass
    if name == "FixedHT":
        return HoeffdingTree(schema, random_seed=seed, grace_period=100, confidence=1e-7)
    if name == "AutoClass":
        return SeededAutoClass(schema=schema, random_seed=seed, configuration_json=str(ROOT / "config/autoclass.json"),
                         base_classifiers=[HoeffdingTree, HoeffdingAdaptiveTree])
    pool = [HoeffdingTree(schema, random_seed=seed, grace_period=g, confidence=1e-7) for g in [50, 100, 200]]
    pool += [HoeffdingAdaptiveTree(schema, random_seed=seed, grace_period=g) for g in [50, 200]]
    if name == "Bandit":
        return RecordedBandit(schema=schema, random_seed=seed, base_classifiers=pool,
                               policy=EpsilonGreedy(epsilon=0.1, burn_in=100), verbose=False)
    return ActiveSuccessiveHalving(schema=schema, random_seed=seed, base_classifiers=pool,
                                      budget=2 * n, eta=2, min_models=1, verbose=False)


def clusterer(name, schema, seed):
    require(name, CLUSTERS, 'cluster model')
    from capymoa.cluster import Clustream, ClusTree, Denstream_with_dbscan
    if name == "CluStream":
        model = Clustream(schema=schema, time_window=1000, max_num_kernels=50, kernel_radi_factor=2)
    elif name == "ClusTree":
        model = ClusTree(schema=schema, horizon=1000, max_height=5, breadth_first_strategy=False)
    else:
        model = Denstream_with_dbscan(schema=schema, horizon=1000, epsilon=0.5, beta=0.2, mu=10,
                  init_points=100, offline_option=2, lambda_option=0.25, speed=100)
    # The wrappers do not expose a seed argument; set the MOA learner seed if supported.
    if hasattr(model.moa_learner, "setRandomSeed"):
        model.moa_learner.setRandomSeed(seed)
    return model
