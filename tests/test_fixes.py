"""Regressions for audit findings: identity, candidate retention and fail-fast config."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from capy_lab.runtime import configure
CONFIG = configure()
from capy_lab.models import classifier, automl, clusterer
from capy_lab.configuration import validate_config
from capy_lab.experiments import signature, compatible
from capy_lab.automl_adapters import ActiveSuccessiveHalving
from lab_support.identity import seal, valid_record, environment
from lab_support.validation import check_table


class FixTests(unittest.TestCase):
    def test_unknown_names_and_unsupported_depth_fail(self):
        for action in [lambda: classifier('River','typo',None,2,42,CONFIG),
                       lambda: classifier('typo','GaussianNB',None,2,42,CONFIG),
                       lambda: automl('typo',None,42,100),
                       lambda: clusterer('typo',None,42)]:
            with self.assertRaises(ValueError): action()
        altered={**CONFIG,'tree_max_depth':2}
        with self.assertRaises(ValueError): validate_config(altered)
        with self.assertRaises(ValueError): classifier('River','HoeffdingTree',None,2,42,altered)

    def test_halving_never_predicts_from_eliminated_candidate(self):
        class Learner:
            def __init__(self,i): self.i=i
            def predict(self,x): return self.i
            def train(self,x): pass
        class Metric:
            def __init__(self,before,after): self.value=before;self.after=after
            def accuracy(self): return self.value
            def update(self,*args): self.value=self.after
        model=ActiveSuccessiveHalving.__new__(ActiveSuccessiveHalving)
        model.active_models=[Learner(i) for i in range(5)]
        model.metrics=[Metric(0,v) for v in [80,70,60,50,10]]
        model.metrics[4].value=90
        for k,v in {'_best_model_idx':4,'_rankings':list(range(5)),'_s':5,'_n':5,'_iterations':0,'_r':1,'_n_rungs':0,'_budget_used':0,'eta':2,'min_models':1,'budget':100,'verbose':False,'observations':0,'training_counts':[0]*5}.items():setattr(model,k,v)
        from types import SimpleNamespace
        model.train(SimpleNamespace(y_index=0))
        self.assertEqual(model._rankings[:model._s],[0,1,2])
        self.assertIn(model._best_model_idx,model._rankings[:model._s])
        self.assertEqual(model.predict(None),0)
        self.assertEqual(sum(model.training_counts),5)

    def test_bandit_preserves_all_candidates_and_reports_actual_policy(self):
        from capy_lab.common import schema_for
        model=automl('Bandit',schema_for(2),42,100)
        for i,metric in enumerate(model.metrics):
            metric.update(0,0)
            model.policy.arm_counts[i]=1
            model.policy.arm_rewards[i]=10 if i==1 else 0
        model._best_model_idx=4
        state=model.get_model_info()
        self.assertEqual(len(state['candidates']),5)
        self.assertEqual(len({c['id'] for c in state['candidates']}),5)
        self.assertEqual(state['best_model_index'],1)
        self.assertEqual(state['upstream_reported_best_index'],4)
        self.assertEqual(state['burn_in_observations'],20)

    def test_runtime_or_record_changes_prevent_cache_reuse(self):
        job={'stage':'classification','dataset':'Phishing','model':'GaussianNB','seed':42,'library':'River',
             'id':'classification__Phishing__42__GaussianNB__River'}
        record=seal({'summary':job},signature(CONFIG,job),2)
        self.assertTrue(compatible(record,CONFIG,job))
        old=copy.deepcopy(environment(2))
        damaged=copy.deepcopy(record);damaged['versions']['river']='wrong'
        self.assertFalse(compatible(damaged,CONFIG,job))
        self.assertEqual(environment(2),old)
        damaged=copy.deepcopy(record);damaged['summary']['seed']=43
        self.assertFalse(compatible(damaged,CONFIG,job))
        damaged=copy.deepcopy(record);damaged['summary']['accuracy']=.999
        self.assertFalse(compatible(damaged,CONFIG,job))
        with patch('lab_support.identity.environment',return_value={**old,'moa_jar_sha256':'changed'}):
            self.assertFalse(compatible(record,CONFIG,job))

    def test_imported_project_source_changes_invalidate_signature(self):
        from lab_support import identity
        before=signature(CONFIG,{'stage':'classification'});original=identity.file_hash
        def modified(path):
            return 'changed' if str(path).endswith('river_lab/data.py') else original(path)
        with patch.object(identity,'file_hash',side_effect=modified):
            self.assertNotEqual(before,signature(CONFIG,{'stage':'classification'}))

    def test_modified_export_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'metrics.csv'
            path.write_text('dataset,seed,accuracy\ntest,42,0.7\n')
            self.assertTrue(check_table(path,[{'dataset':'test','seed':42,'accuracy':.7}],['dataset','seed']))
            with self.assertRaises(ValueError):check_table(path,[{'dataset':'test','seed':42,'accuracy':.9}],['dataset','seed'])

    def test_overlay_is_loaded_and_random_state_is_explicit(self):
        import jpype
        helper=jpype.JClass('lab4.SeededRandom')
        source=jpype.JClass('java.util.Random')
        helper.use(source(42));first=[helper.random() for _ in range(5)]
        helper.use(source(42));self.assertEqual(first,[helper.random() for _ in range(5)])
        helper.use(source(43));self.assertNotEqual(first,[helper.random() for _ in range(5)])


if __name__=='__main__':unittest.main()
