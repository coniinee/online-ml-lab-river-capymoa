"""Repeat complete AutoClass streams in fresh processes with fixed reviewed sources."""
from pathlib import Path
import sys,json,subprocess,argparse,random,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/fixes/autoclass_reproducibility'

def worker(ds,seed,repeat):
 from capy_lab.runtime import configure
 cfg=configure()
 import numpy as np
 random.seed(seed);np.random.seed(seed)
 from capy_lab import experiments
 from unittest.mock import patch
 from scripts.audit_experiments import classification_metrics, compare_mapping
 from capy_lab.common import classification_data
 captured=[];factory=experiments.automl
 class Capture:
  def __init__(self,learner):self.learner=learner
  def __getattr__(self,name):return getattr(self.learner,name)
  def predict(self,instance):
   value=self.learner.predict(instance);captured.append(0 if value is None else int(value));return value
  def train(self,instance):return self.learner.train(instance)
 job={'stage':'automl','dataset':ds,'model':'AutoClass','library':'CapyMOA','seed':seed}
 with patch.object(experiments,'automl',side_effect=lambda *args:Capture(factory(*args))):
  result=experiments.classification_job(job,cfg)
 _,truth,_=classification_data(ds)
 for point in result['curves']:
  end=point['n_seen'];metrics=classification_metrics(truth[:end],captured[:end])
  assert not compare_mapping(metrics,point,metrics)
 data={'dataset':ds,'seed':seed,'repeat':repeat,'prediction_sha256':result['prediction_sha256'],
       'metrics':{k:result['summary'][k] for k in ['accuracy','macro_f1','kappa']},
       'selection_sha256':hashlib.sha256(json.dumps([p['selection_state'] for p in result['curves']],sort_keys=True).encode()).hexdigest(),
       'final_state':json.loads(result['curves'][-1]['selection_state']),
       'all_checkpoint_metrics_independently_verified':True}
 (OUT/f'{ds}__{seed}__{repeat}.json').write_text(json.dumps(data,indent=2)+'\n')
 print(ds,seed,repeat,data['metrics']['accuracy'],data['prediction_sha256'],flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset');p.add_argument('--seed',type=int);p.add_argument('--repeat',type=int);p.add_argument('--against-formal',action='store_true');a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 if a.dataset:worker(a.dataset,a.seed,a.repeat)
 else:
  rows=[]
  for ds in ['Elec2','SEARecurring']:
   for seed in [42,43,44]:
    count=1 if a.against_formal else 2
    for rep in range(1,count+1):
     subprocess.run([sys.executable,str(Path(__file__).resolve()),'--dataset',ds,'--seed',str(seed),'--repeat',str(rep)],check=True,cwd=ROOT)
     rows.append(json.loads((OUT/f'{ds}__{seed}__{rep}.json').read_text()))
    selected=[r for r in rows if r['dataset']==ds and r['seed']==seed]
    assert len({r['prediction_sha256'] for r in selected})==1
    assert len({r['selection_sha256'] for r in selected})==1
    if a.against_formal:
     formal=json.loads((ROOT/'results/capymoa/runs'/f'automl__{ds}__{seed}__AutoClass__CapyMOA.json').read_text())
     assert selected[0]['prediction_sha256']==formal['prediction_sha256']
     assert selected[0]['metrics']=={k:formal['summary'][k] for k in selected[0]['metrics']}
  target='formal_reproduction.json' if a.against_formal else 'preflight_reproduction.json'
  (ROOT/'results/fixes'/target).write_text(json.dumps({'all_prediction_and_selection_repeats_equal':True,'cases':rows},indent=2)+'\n')
