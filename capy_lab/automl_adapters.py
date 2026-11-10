"""Minimal fixes for CapyMOA 0.15.1; preserve upstream training/search policies."""
import hashlib
import json
import math
import threading

from capymoa.automl import AutoClass, BanditClassifier, SuccessiveHalvingClassifier
from .runtime import ROOT

_AUTOCLASS_LOCK = threading.RLock()


def candidate_state(models, metrics, counts, active):
    return [{'id': f'candidate_{i}', 'index': i, 'model': type(model).__name__,
             'parameters': str(model.moa_learner.getOptions().getAsCLIString()),
             'accuracy_percent': float(metrics[i].accuracy()) if counts[i] else None,
             'training_updates': counts[i], 'active': i in active}
            for i, model in enumerate(models)]


class RecordedBandit(BanditClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observations = 0

    def train(self, instance):
        super().train(instance)
        self.observations += 1

    def get_model_info(self):
        stats = self.policy.get_arm_stats()
        index = self.policy.get_best_arm_idx(range(len(self.active_models)))
        return {'schema': 2, 'method': 'Bandit', 'observations': self.observations,
                'total_models': len(self.active_models), 'best_model_index': index,
                'prediction_candidate_id': f'candidate_{index}',
                'upstream_reported_best_index': self._best_model_idx,
                'candidates': candidate_state(self.active_models, self.metrics, stats['counts'], range(len(self.active_models))),
                'policy': stats, 'burn_in_pulls': self.policy.burn_in,
                'burn_in_observations': math.ceil(self.policy.burn_in/len(self.active_models)),
                'reward_definition': 'Cumulative candidate accuracy in percent per pull; upstream policy unchanged.'}


class ActiveSuccessiveHalving(SuccessiveHalvingClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_counts = [0] * len(self.active_models)
        self.observations = 0

    def _ensure_active(self):
        active = self._rankings[:self._s]
        if self._best_model_idx not in active:
            self._best_model_idx = max(active, key=lambda i: self.metrics[i].accuracy())

    def train(self, instance):
        for i in self._rankings[:self._s]: self.training_counts[i] += 1
        super().train(instance)
        self._ensure_active()
        self.observations += 1

    def predict(self, instance):
        self._ensure_active()
        return super().predict(instance)

    def predict_proba(self, instance):
        self._ensure_active()
        return super().predict_proba(instance)

    def get_model_info(self):
        self._ensure_active()
        return {'schema': 2, 'method': 'SuccessiveHalving', 'observations': self.observations,
                'active_models': self._s, 'total_models': self._n, 'active_indices': self._rankings[:self._s],
                'current_rung': self._n_rungs, 'total_budget': self.budget,
                'budget_used': self._budget_used, 'budget_left': self.budget-self._budget_used,
                'iterations_in_current_rung': self._iterations,
                'actual_model_updates': sum(self.training_counts),
                'best_model_index': self._best_model_idx,
                'prediction_candidate_id': f'candidate_{self._best_model_idx}',
                'candidates': candidate_state(self.active_models, self.metrics, self.training_counts, self._rankings[:self._s])}


class SeededAutoClass(AutoClass):
    def __init__(self, *args, random_seed=0, **kwargs):
        import jpype
        settings = json.loads((ROOT/'config/autoclass.json').read_text())
        if settings['numberOfCores'] != 1:
            raise ValueError('Deterministic AutoClass requires numberOfCores=1')
        self._random_source = jpype.JClass('java.util.Random')(random_seed)
        self._seeded = jpype.JClass('lab4.SeededRandom')
        self.observations = 0
        with _AUTOCLASS_LOCK:
            self._seeded.use(self._random_source)
            super().__init__(*args, random_seed=random_seed, **kwargs)
        source = str(self.moa_learner.getClass().getProtectionDomain().getCodeSource().getLocation())
        if 'automl-seeded.jar' not in source:
            raise RuntimeError('AutoClass loaded before the reviewed overlay; restart the Python kernel')

    def predict(self, instance):
        with _AUTOCLASS_LOCK:
            self._seeded.use(self._random_source)
            return super().predict(instance)

    def predict_proba(self, instance):
        with _AUTOCLASS_LOCK:
            self._seeded.use(self._random_source)
            return super().predict_proba(instance)

    def train(self, instance):
        with _AUTOCLASS_LOCK:
            self._seeded.use(self._random_source)
            super().train(instance)
            self.observations += 1

    def get_model_info(self):
        candidates = []
        for i, algorithm in enumerate(self.moa_learner.ensemble):
            cli = str(algorithm.classifier.getOptions().getAsCLIString())
            name = str(algorithm.algorithm)
            score = float(algorithm.performanceMeasure)
            candidates.append({'slot': i, 'id': hashlib.sha256((name+cli).encode()).hexdigest()[:16],
                               'model': name, 'parameters': cli,
                               'search_parameters': {str(p.getParameter()): str(p.getCLIValueString()) for p in algorithm.parameters},
                               'performance': score if math.isfinite(score) else None})
        return {'schema': 2, 'method': 'AutoClass', 'observations': self.observations,
                'best_model_index': int(self.moa_learner.bestModel), 'candidates': candidates,
                'total_models': len(candidates), 'random_seed': self.random_seed,
                'random_source': 'Per-learner java.util.Random via reviewed AutoML overlay; serial execution.'}
