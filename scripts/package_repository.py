"""Package the formal source materials without extra delivery manifests.

Does not train models, generate PDFs, export notebook HTML or publish a repository.
"""
from pathlib import Path
import hashlib,json,tempfile,zipfile

ROOT=Path(__file__).resolve().parents[1]
SKIP={'.git','.cache','__pycache__','.ipynb_checkpoints','__MACOSX','verification'}
ROOT_EXCLUSIONS={'PAYLOAD_MANIFEST.json','SHA256SUMS.txt','SUBMISSION_CHECKLIST.md'}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    files=[]
    for p in ROOT.rglob('*'):
        relative=p.relative_to(ROOT)
        if any(part in SKIP or part.startswith('.venv') for part in relative.parts):continue
        if len(relative.parts)==1 and p.name in ROOT_EXCLUSIONS:continue
        if p.is_symlink() or not p.is_file():continue
        if p.name=='.DS_Store' or p.name.startswith('._') or p.suffix in ['.pyc','.tmp','.zip']:continue
        assert str(relative).isascii(),relative
        files.append(p)
    files.sort();output=ROOT.parent/(ROOT.name+'.zip');temporary=output.with_suffix('.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in files:archive.write(p,ROOT.name+'/'+p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None and len(archive.namelist())==len(files)
        with tempfile.TemporaryDirectory(prefix='lab4-formal-',dir='/private/tmp' if Path('/private/tmp').exists() else None) as work:
            archive.extractall(work)
            for p in files:assert sha(Path(work)/ROOT.name/p.relative_to(ROOT))==sha(p),p
    temporary.replace(output)
    print(json.dumps({'archive':str(output),'files':len(files),'MiB':round(output.stat().st_size/2**20,2),
                      'sha256':sha(output),'CRC_and_default_extraction_verified':True},indent=2))

if __name__=='__main__':main()
