"""Pin official metadata and download only the chosen models/SFT subset to outputs."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import time

ROOT=Path(__file__).resolve().parents[4]
PHASE=ROOT/'research/experiments/phase5_ce_reference_distillation'
OUT=ROOT/'outputs/phase5/modern_assets'

def get(url):
    with urllib.request.urlopen(url,timeout=45) as r:return r.read()

def main():
    from huggingface_hub import snapshot_download
    OUT.mkdir(parents=True,exist_ok=True)
    saved=PHASE/'modern_assets.json'
    manifest=json.loads(saved.read_text()) if saved.exists() else {'models':{},'download_complete':False}
    for name in ['Qwen/Qwen3-0.6B-Base','Qwen/Qwen3-1.7B-Base']:
        if name in manifest['models']:continue
        meta=json.loads(get('https://huggingface.co/api/models/'+name))
        entry={'repo':name,'revision':meta['sha'],'license':meta.get('cardData',{}).get('license'),
               'parameters':meta.get('safetensors'),'local_dir':str((OUT/name.split('/')[-1]).relative_to(ROOT))}
        manifest['models'][name]=entry
        print(json.dumps(entry),flush=True)
    ds=json.loads(get('https://huggingface.co/api/datasets/HuggingFaceTB/smoltalk2'+
        ('/revision/'+manifest['data']['revision'] if 'data' in manifest else '')))
    paths=[r['rfilename'] for r in ds['siblings'] if 'smol_summarize' in r['rfilename'] and 'no_think' in r['rfilename']]
    if not paths: raise RuntimeError('expected summarization subset not found; do not download entire mixture')
    manifest['data']={'repo':'HuggingFaceTB/smoltalk2','revision':ds['sha'],'config':'SFT',
        'split':'smoltalk_smollm3_smol_summarize_no_think','files':paths,'card':ds.get('cardData',{}),
        'local_dir':str((OUT/'smoltalk2').relative_to(ROOT))}
    (PHASE/'modern_assets.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def bounded_download(**kwargs):
        for attempt in range(1,5):
            try:return snapshot_download(**kwargs,max_workers=2)
            except Exception as exc:
                print(json.dumps({'download_attempt':attempt,'error_type':type(exc).__name__,
                                  'repo':kwargs['repo_id'],'resume_incomplete':True}),flush=True)
                if attempt==4:raise
                time.sleep(5)
    bounded_download(repo_id=manifest['data']['repo'],repo_type='dataset',revision=ds['sha'],
        local_dir=OUT/'smoltalk2',allow_patterns=paths+['README.md','LICENSE'])
    for entry in manifest['models'].values():
        bounded_download(repo_id=entry['repo'],revision=entry['revision'],local_dir=ROOT/entry['local_dir'],
            allow_patterns=['*.json','*.safetensors','*.txt','*.md','LICENSE'])
    manifest['download_complete']=True
    (PHASE/'modern_assets.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('OFFICIAL_ASSETS_DOWNLOADED',flush=True)

if __name__=='__main__':main()
