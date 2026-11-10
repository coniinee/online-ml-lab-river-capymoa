"""Verify distributed evidence without the earlier, optional audit archives.

Default checks validate the supplied historical results on any supported Python
environment; --strict-runtime additionally requires the original runtime identity.
No model is fitted, no experiment files are overwritten, and no PDF is created.
"""
from pathlib import Path
import argparse,gzip,hashlib,json,subprocess,sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--require-manifest',action='store_true')
    group.add_argument('--skip-manifest',action='store_true',help='Only for a working copy whose results were intentionally regenerated')
    parser.add_argument('--strict-runtime',action='store_true')
    parser.add_argument('--river-python',type=Path,help='River interpreter for strict checking; defaults to .venv/bin/python')
    args=parser.parse_args()
    if args.strict_runtime:
        from capy_lab.runtime import configure
        cfg=configure()
    manifest_path=ROOT/'PAYLOAD_MANIFEST.json'
    if args.require_manifest:assert manifest_path.is_file(),'Missing PAYLOAD_MANIFEST.json'
    file_count=0
    if manifest_path.exists() and not args.skip_manifest:
        manifest=json.loads(manifest_path.read_text())
        for name,entry in manifest['files'].items():
            p=ROOT/name
            assert p.is_file() and p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],name
            file_count+=1
    from lab_support.identity import record_digest
    from lab_support.validation import validate_exports,check_table
    import pandas as pd
    parts={part:[json.loads(p.read_text()) for p in sorted((ROOT/f'results/{part}/runs').glob('*.json'))] for part in ['river','capymoa']}
    assert len(parts['river'])==168 and len(parts['capymoa'])==147
    records=parts['river']+parts['capymoa']
    assert all(r['record_sha256']==record_digest(r) for r in records)
    validate_exports(ROOT/'results/river',parts['river'],1)
    validate_exports(ROOT/'results/capymoa',parts['capymoa'],2)
    data_manifest=json.loads((ROOT/'data/manifest.json').read_text())
    for item in data_manifest['datasets']:
        with gzip.open(ROOT/item['cache'],'rb') as stream:raw=stream.read()
        assert hashlib.sha256(raw).hexdigest()==item['sha256_uncompressed'],item['dataset']
        assert len(raw.splitlines())==item['n_samples'],item['dataset']
    drifts=pd.read_csv(ROOT/'results/river/drift_summary.csv')
    events=pd.read_csv(ROOT/'results/river/drift_events.csv')
    assert drifts['mode'].value_counts().to_dict()=={'monitor':72,'reset':72}
    assert not drifts.duplicated(['dataset','seed','mode','detector']).any()
    for row in drifts.to_dict('records'):
        selected=events[(events.dataset==row['dataset'])&(events.seed==row['seed'])&(events['mode']==row['mode'])&(events.detector==row['detector'])]
        positions=selected.position.tolist()
        assert positions==json.loads(row['alarm_positions']) and len(positions)==row['n_alarms']
    from river_lab.evaluation import drift_record
    config=json.loads((ROOT/'config/river.json').read_text())
    monitors=[]
    for row in drifts[drifts['mode']=='monitor'].to_dict('records'):
        item=next(d for d in data_manifest['datasets'] if d['dataset']==row['dataset'])
        selected=events[(events.dataset==row['dataset'])&(events.seed==row['seed'])&(events['mode']=='monitor')&(events.detector==row['detector'])]
        expected=drift_record(item,row['seed'],row['detector'],'monitor',selected.to_dict('records'),config)
        baseline=next(r['summary'] for r in parts['river'] if r['summary']['dataset']==row['dataset'] and r['summary']['seed']==row['seed'] and r['summary']['model']=='HoeffdingTree' and r['summary']['mode']=='baseline')
        expected.update({k:baseline[k] for k in ['mae','rmse','r2','accuracy','macro_f1','kappa'] if k in baseline})
        assert row['detector_only_runtime_s']>=0 and row['detector_only_mean_latency_us']>=0
        monitors.append(expected)
    reset=[r['drift_summary'] for r in parts['river'] if r['summary']['mode']=='reset']
    check_table(ROOT/'results/river/drift_summary.csv',
                [*monitors,*reset],
                ['dataset','seed','mode','detector'])
    checkpoints=0
    for r in parts['capymoa']:
        s=r['summary']
        if s['stage']!='automl':continue
        for point in r['curves']:
            state=json.loads(point['selection_state']);assert state['observations']==point['n_seen']
            if s['model']=='FixedHT':continue
            candidates=state['candidates'];assert candidates
            if s['model'] in ['Bandit','SuccessiveHalving']:
                assert len(candidates)==5 and len({c['id'] for c in candidates})==5
                chosen=next(c for c in candidates if c['index']==state['best_model_index'])
                assert chosen['active'] and chosen['id']==state['prediction_candidate_id']
            else:assert state['random_seed']==s['seed'] and 0<=state['best_model_index']<len(candidates)
            checkpoints+=1
    assert checkpoints==900
    reproduction=json.loads((ROOT/'results/fixes/formal_reproduction.json').read_text())
    assert reproduction['all_prediction_and_selection_repeats_equal'] and len(reproduction['cases'])==6
    for row in reproduction['cases']:
        r=next(r for r in parts['capymoa'] if r['summary']['stage']=='automl' and r['summary']['model']=='AutoClass' and r['summary']['dataset']==row['dataset'] and r['summary']['seed']==row['seed'])
        assert r['prediction_sha256']==row['prediction_sha256']
        assert hashlib.sha256(json.dumps([p['selection_state'] for p in r['curves']],sort_keys=True).encode()).hexdigest()==row['selection_sha256']
    patch=json.loads((ROOT/'vendor/automl-seeded.json').read_text())
    assert sha(ROOT/'vendor/automl-seeded.jar')==patch['overlay_sha256']
    for name,expected in patch['sources'].items():assert sha(ROOT/name)==expected,name
    if args.strict_runtime:
        from capy_lab.experiments import compatible
        assert all(compatible(r,cfg) for r in parts['capymoa'])
        code="import json; from pathlib import Path; from river_lab.evaluation import fingerprint; from lab_support.identity import valid_record; c=json.loads(Path('config/river.json').read_text());m=json.loads(Path('data/manifest.json').read_text());s=fingerprint(c,m);r=[json.loads(p.read_text()) for p in Path('results/river/runs').glob('*.json')];assert len(r)==168 and all(valid_record(x,s,1) for x in r)"
        subprocess.run([str(args.river_python or ROOT/'.venv/bin/python'),'-c',code],cwd=ROOT,check=True)
    print(json.dumps({'files_verified_against_distribution_manifest':file_count,'sealed_model_records':315,
                      'raw_records_and_csv_exports_match':True,'ordered_data_caches_verified':8,
                      'monitor_replays':72,'reset_comparisons':72,'alarm_position_lists_checked':144,
                      'automl_selection_checkpoints':checkpoints,'existing_autoclass_reproduction_cases_matched':6,
                      'java_overlay_and_sources_verified':True,'strict_current_runtime_check':args.strict_runtime,
                      'model_training_performed':False,'all_checks_passed':True},indent=2))

if __name__=='__main__':main()
