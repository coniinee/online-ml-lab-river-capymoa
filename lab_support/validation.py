"""Compare exported tables with sealed raw evidence before presenting results."""
import json
import math
import numbers
import pandas as pd


def equivalent(a, b):
    if isinstance(b, (list, dict)):
        try: return json.loads(a) == b
        except (TypeError, ValueError): return a == b
    if b is None: return a is None or bool(pd.isna(a))
    if isinstance(b, numbers.Number):
        return isinstance(a, numbers.Number) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-11)
    return a == b


def check_table(path, expected, keys):
    frame = pd.read_csv(path)
    if len(frame) != len(expected) or frame.duplicated(keys).any():
        raise ValueError(f'Export coverage/duplicate mismatch: {path}')
    actual = {tuple(r[k] for k in keys):r for r in frame.to_dict('records')}
    for row in expected:
        key = tuple(row[k] for k in keys)
        if key not in actual: raise ValueError(f'Missing exported key {key}: {path}')
        for name, value in row.items():
            if name not in actual[key] or not equivalent(actual[key][name], value):
                raise ValueError(f'Export disagrees with raw evidence: {path}, {key}, {name}')
    return True


def validate_exports(folder, records, part):
    if part == 1:
        for task in ['regression','classification']:
            expected=[r['summary'] for r in records if r['summary']['task']==task and r['summary']['mode']=='baseline']
            check_table(folder/(task+'_metrics.csv'),expected,['dataset','model','seed','mode','detector'])
        check_table(folder/'reset_metrics.csv',[r['summary'] for r in records if r['summary']['mode']=='reset'],['dataset','model','seed','mode','detector'])
        check_table(folder/'learning_curves.csv',[p for r in records for p in r['curves']],['dataset','model','seed','mode','detector','n_seen'])
        check_table(folder/'recovery_windows.csv',[p for r in records for p in r['windows']],['dataset','model','seed','mode','detector','change_position','window'])
    else:
        for stage in ['classification','clustering','automl']:
            selected=[r for r in records if r['summary']['stage']==stage]
            check_table(folder/(stage+'_metrics.csv'),[r['summary'] for r in selected],['dataset','model','library','seed'])
            check_table(folder/(stage+'_curves.csv'),[{**{k:r['summary'][k] for k in ['dataset','model','library','seed']},**p} for r in selected for p in r['curves']],['dataset','model','library','seed','n_seen'])
    return True
