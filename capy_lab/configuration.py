"""Fail early for misspelled jobs and unsupported comparison settings."""
FAMILIES = {'HoeffdingTree','AdaptiveRandomForest','LinearLogistic','GaussianNB'}
AUTOML = {'FixedHT','Bandit','SuccessiveHalving','AutoClass'}
CLUSTERS = {'CluStream','ClusTree','DenStream'}
LIBRARIES = {'River','CapyMOA'}


def require(value, choices, label):
    if value not in choices:
        raise ValueError(f'Unknown {label}: {value!r}; expected one of {sorted(choices)}')


def validate_config(config):
    for key, choices in [('families',FAMILIES),('automl_models',AUTOML),('cluster_models',CLUSTERS)]:
        values = config[key]
        if not values or len(values) != len(set(values)):
            raise ValueError(f'{key} must be nonempty and contain no duplicates')
        for value in values: require(value, choices, key)
    if config.get('tree_max_depth') is not None:
        raise ValueError('tree_max_depth must be null: CapyMOA has no equivalent limit for this paired experiment')
    for key in ['checkpoint_every','rolling_window','forest_trees','tree_grace_period','tree_memory_mib','cluster_calibration','cluster_macro_k','cluster_refresh','cluster_eval_every','cluster_synthetic_n','worker_timeout_s']:
        if not isinstance(config[key],int) or isinstance(config[key],bool) or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if config['cluster_calibration'] >= config['cluster_synthetic_n']:
        raise ValueError('Calibration must leave scored synthetic observations')
    for key in ['seeds','classification_datasets','cluster_datasets','automl_datasets']:
        if not config[key] or len(config[key]) != len(set(config[key])):
            raise ValueError(f'{key} must contain unique values')
    if any(not isinstance(seed,int) or isinstance(seed,bool) for seed in config['seeds']):
        raise ValueError('seeds must contain integers')
    classification = {'Elec2','Phishing','ImageSegments','SEARecurring'}
    for key, names in [('classification_datasets',classification),('automl_datasets',classification),('cluster_datasets',classification|{'MovingGaussians'})]:
        for value in config[key]: require(value,names,key)
    return config
