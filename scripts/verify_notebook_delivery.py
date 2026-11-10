"""Check saved executed notebook outputs directly against the current CSVs; no HTML export."""
from pathlib import Path
import argparse,ast,base64,collections,hashlib,json,re,sys
ROOT=Path(__file__).resolve().parents[1]
import nbformat,pandas as pd
from bs4 import BeautifulSoup

def canonical(table):return re.sub(r'\s+',' ',table.get_text(' ',strip=True)).strip()
def check(part):
 label='river' if part==1 else 'capymoa';name='river_steps_1_5' if part==1 else 'capymoa_steps_7_10'
 folder=ROOT/'results'/label
 nb=nbformat.read(ROOT/'notebooks'/f'{name}.ipynb',as_version=4);nbformat.validate(nb)
 code=[c for c in nb.cells if c.cell_type=='code'];assert all(c.execution_count is not None for c in code)
 assert not any(o.output_type=='error' for c in code for o in c.outputs)
 saved_tables=collections.Counter();figures=[]
 for cell in code:
  for output in cell.outputs:
   data=output.get('data',{})
   if 'text/html' in data:
    saved_tables.update(canonical(t) for t in BeautifulSoup(data['text/html'],'html.parser').find_all('table'))
   if 'image/png' in data:figures.append(hashlib.sha256(base64.b64decode(data['image/png'])).hexdigest())
 rendered=saved_tables
 sources=collections.Counter(hashlib.sha256(p.read_bytes()).hexdigest() for p in (folder/'figures').glob('*.png'))
 assert collections.Counter(figures)==sources
 checked=[]
 def table(frame,title):
  with pd.option_context('display.float_format',lambda x:f'{x:.4f}','display.max_rows',100,'display.max_columns',20):
   expected=canonical(BeautifulSoup(frame._repr_html_(),'html.parser').find('table'))
  assert rendered[expected]>0, title+' is stale or absent from notebook outputs'
  checked.append(title)
 def read(name):return pd.read_csv(folder/name)
 if part==1:
  for task,metrics in [('regression',['mae','rmse','r2']),('classification',['accuracy','macro_f1','kappa'])]:
   df=read(task+'_metrics.csv')
   cols=metrics+['mean_latency_us','p95_latency_us','total_runtime_s','sampled_peak_process_rss_mb']
   table(df.groupby(['dataset','model'],sort=False)[cols].mean().reset_index(),task+' main results')
   drift=read('drift_summary.csv');d=drift[(drift.task==task)&(drift['mode']=='monitor')]
   table(d.groupby(['dataset','detector'],sort=False)[['n_alarms','matched_changes','missed_changes','unmatched_alarms','mean_delay','detector_only_runtime_s']].mean().reset_index(),task+' drift results')
   effects=read('reset_effect_summary.csv');table(effects[effects.task==task].dropna(axis=1,how='all'),task+' reset results')
  assert 'python -m unittest discover -s tests -p test_protocol.py -v' in '\n'.join(c.source for c in nb.cells)
 else:
  df=read('classification_metrics.csv')
  table(df.groupby(['dataset','model','library'],sort=False)[['accuracy','macro_f1','kappa','mean_latency_us','p95_latency_us','total_runtime_s','sampled_peak_process_rss_mib']].mean().reset_index(),'cross-library classification')
  df=read('classification_pairs.csv');table(df.groupby(['dataset','model'],sort=False)[['accuracy_difference_pp','capy_to_river_latency_ratio']].mean().reset_index(),'paired differences')
  df=read('clustering_metrics.csv');table(df.groupby(['dataset','model'],sort=False)[['mean_window_ari','mean_window_silhouette','mean_window_coverage','mean_window_mse','mean_latency_us','total_runtime_s','final_micro_clusters']].mean().reset_index(),'clustering')
  df=read('automl_metrics.csv');table(df.groupby(['dataset','model'],sort=False)[['accuracy','macro_f1','kappa','mean_latency_us','total_runtime_s']].mean().reset_index(),'AutoML')
 assert [c.execution_count for c in code]==list(range(1,len(code)+1)), 'Execute the whole notebook in order'
 assert not re.search(r'[\u3400-\u9fff]',json.dumps(nb,ensure_ascii=False)), 'Non-English prose remains'
 links=0
 for cell in nb.cells:
  if cell.cell_type!='markdown':continue
  for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',cell.source):
   if target.startswith(('https://','http://','#')):continue
   assert (ROOT/'notebooks'/target.split('#')[0]).exists(),target
   links+=1
 controls=[c for c in code if 'RERUN_EXPERIMENTS' in c.source]
 assert len(controls)==1
 assigned=next(n for n in ast.parse(controls[0].source).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RERUN_EXPERIMENTS' for t in n.targets))
 assert isinstance(assigned.value,ast.Constant) and assigned.value.value is False
 demo=next(c for c in code if 'DEMO_N = 100' in c.source)
 outputs=[ast.literal_eval(o.data['text/plain']) for o in demo.outputs if o.get('data',{}).get('text/plain','').startswith("{'demo_only'")]
 assert len(outputs)==1 and outputs[0]['demo_only'] is True and outputs[0]['examples']==100
 result={'part':part,'executed_code_cells':len(code),'notebook_errors':0,'saved_output_tables_checked':sum(saved_tables.values()),'embedded_figures_match_current_png_files':len(figures),'primary_tables_match_current_csv':checked,'offline_figure_dependencies':0,'source_links_checked':links,'formal_retraining_default':False,'bounded_demo_examples':100}
 out=ROOT/'verification';out.mkdir(exist_ok=True)
 (out/f'{label}_notebook.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--part',type=int,choices=[1,2],required=True);check(p.parse_args().part)
