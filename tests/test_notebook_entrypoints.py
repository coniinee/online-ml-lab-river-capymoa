"""Check the runnable notebook controls and repository navigation without fitting."""
from pathlib import Path
import ast,json,types,unittest

ROOT=Path(__file__).resolve().parents[1]

def notebook(name):
    return json.loads((ROOT/'notebooks'/f'{name}.ipynb').read_text())

def source(cell):return ''.join(cell['source']) if isinstance(cell['source'],list) else cell['source']

def requested_commands(name):
    cell=next(c for c in notebook(name)['cells'] if c['cell_type']=='code' and 'RERUN_EXPERIMENTS' in source(c))
    node=next(n for n in ast.parse(source(cell)).body if isinstance(n,ast.If) and 'RERUN_EXPERIMENTS' in ast.unparse(n.test))
    calls=[]
    def capture(args,**kwargs):
        if kwargs.get('check') is not True:raise AssertionError('Subprocess failures must propagate')
        calls.append(args)
    namespace={'RERUN_EXPERIMENTS':True,'sys':types.SimpleNamespace(executable='current-python'),
               'subprocess':types.SimpleNamespace(run=capture)}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<notebook-run-control>','exec'),namespace)
    return calls

class NotebookEntryTests(unittest.TestCase):
    def test_river_rerun_uses_existing_cache_then_analysis(self):
        self.assertEqual(requested_commands('river_steps_1_5'),[
            ['current-python','scripts/run_river.py','run','--resume'],
            ['current-python','scripts/run_river.py','analyze']])

    def test_capymoa_control_keeps_formal_cli_semantics(self):
        self.assertEqual(requested_commands('capymoa_steps_7_10'),[
            ['current-python','scripts/run_capy.py','all','--resume'],
            ['current-python','scripts/run_capy.py','analyze']])

    def test_environment_location_is_not_a_notebook_assertion(self):
        for name in ['river_steps_1_5','capymoa_steps_7_10']:
            startup=next(c for c in notebook(name)['cells'] if c['cell_type']=='code')
            for node in ast.walk(ast.parse(source(startup))):
                if isinstance(node,ast.Assert):self.assertNotIn('sys.prefix',ast.unparse(node.test))

    def test_notebooks_link_to_the_actual_implementation_and_guide(self):
        for name,implementation,entry in [
            ('river_steps_1_5','river_lab/evaluation.py','scripts/run_river.py'),
            ('capymoa_steps_7_10','capy_lab/experiments.py','scripts/run_capy.py')]:
            intro=source(notebook(name)['cells'][0])
            for target in [implementation,entry,'REPRODUCIBILITY.md']:
                self.assertIn(f'(../{target})',intro)
                self.assertTrue((ROOT/target).is_file(),target)

if __name__=='__main__':unittest.main()
