"""Part 2 protocol checks: causal transforms and labels hidden from predictors."""

import unittest
from unittest.mock import patch
import numpy as np

from capy_lab.runtime import configure
CONFIG=configure()
from capy_lab.common import CausalScaler
from capy_lab.experiments import classification_job, clustering_job


class CapyProtocolTests(unittest.TestCase):
    def test_scaling_uses_only_observed_history(self):
        scaler=CausalScaler(2)
        np.testing.assert_array_equal(scaler.transform(np.array([2.,4.])),[2.,4.])
        self.assertEqual(scaler.n,0)
        scaler.update(np.array([2.,4.]));scaler.update(np.array([4.,8.]))
        np.testing.assert_allclose(scaler.transform(np.array([3.,6.])),[0.,0.])
        self.assertEqual(scaler.n,2)

    def test_true_label_not_passed_to_prediction(self):
        calls=[]
        class Learner:
            def predict(self,instance):
                calls.append(('predict',instance.y_index));return 1
            def train(self,instance):calls.append(('train',instance.y_index))
        x=np.array([[0.],[1.],[2.]]);y=np.array([1,0,1])
        meta={'class_labels':[False,True],'feature_names':['x'],'dense_sha256':'test','known_drift_positions':[]}
        job={'stage':'classification','library':'CapyMOA','model':'GaussianNB','dataset':'test','seed':42}
        with patch('capy_lab.experiments.classification_data',return_value=(x,y,meta)),patch('capy_lab.experiments.classifier',return_value=Learner()):
            result=classification_job(job,CONFIG)
        self.assertEqual(calls,[('predict',0),('train',1),('predict',0),('train',0),('predict',0),('train',1)])
        self.assertAlmostEqual(result['summary']['accuracy'],2/3)

    def test_cluster_training_never_sees_evaluation_labels(self):
        trained=[]
        class Learner:
            def train(self,instance):
                self.assert_dummy(instance)
            def assert_dummy(self,instance):
                if instance.y_index!=0:raise AssertionError('Cluster labels leaked')
                trained.append(instance.x.tolist())
        x=np.array([[float(i),float(i%3)] for i in range(12)])
        config={**CONFIG,'cluster_calibration':3,'cluster_eval_every':4,'cluster_refresh':4,'rolling_window':4}
        job={'stage':'clustering','library':'CapyMOA','model':'test','dataset':'test','seed':42}
        def run(labels):
            with patch('capy_lab.experiments.clustering_data',return_value=(x,labels,{'dense_sha256':'test'})),patch('capy_lab.experiments.clusterer',return_value=Learner()),patch('capy_lab.experiments.macro_centers',return_value=(np.array([[-1.,-1.],[1.,1.]]),2)):
                return clustering_job(job,config)
        first=run(np.arange(12)%3);second=run(np.arange(12)%2)
        self.assertEqual(first['snapshots'],second['snapshots'])
        self.assertEqual(trained[:12],trained[12:])


if __name__=='__main__':unittest.main()
