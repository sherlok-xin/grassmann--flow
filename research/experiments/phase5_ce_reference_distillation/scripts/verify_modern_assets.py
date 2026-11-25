"""Local workstation: hash downloaded weights against pinned official LFS metadata."""
import json
import time
import urllib.request
from pathlib import Path
from prepare_modern import sha,ROOT,PHASE


def main():
    path=PHASE/'modern_assets.json';start=time.monotonic()
    while True:
        try:m=json.loads(path.read_text())
        except (FileNotFoundError,json.JSONDecodeError):m={}
        if m.get('download_complete'):break
        if time.monotonic()-start>24*3600:raise TimeoutError('asset download incomplete; no training release')
        time.sleep(30)
    verified=[]
    for e in m['models'].values():
        url=f"https://huggingface.co/api/models/{e['repo']}/revision/{e['revision']}?blobs=true"
        with urllib.request.urlopen(url,timeout=90) as r:meta=json.load(r)
        if meta['sha']!=e['revision']:raise ValueError('model revision drift')
        weights=[s for s in meta['siblings'] if s['rfilename'].endswith('.safetensors')]
        if not weights:raise ValueError('no pinned weight files')
        for w in weights:
            f=ROOT/e['local_dir']/w['rfilename'];observed=sha(f);lfs=w.get('lfs',{})
            expected=lfs.get('sha256') or lfs.get('oid')
            if not expected or observed!=expected:raise ValueError('weight checksum differs from official LFS metadata')
            verified.append(dict(repo=e['repo'],revision=e['revision'],path=str(f.relative_to(ROOT)),sha256=observed,bytes=f.stat().st_size))
    out=PHASE/'modern_assets_verified.json'
    out.write_text(json.dumps(dict(status='VERIFIED',assets_manifest_sha256=sha(path),weights=verified,
        no_training_or_evaluation=True),indent=2)+'\n')
    print(out.read_text(),flush=True)


if __name__=='__main__':main()
