"""Download exact official base revisions locally; remote training reads shared files."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[4]
DEST = ROOT / 'outputs/phase4a_modern/model_cache'
ART = Path(__file__).resolve().parents[1]
MODEL_REVISIONS = {
    'SmolLM2-135M': '93efa2f097d58c2a74874c7e644dbc9b0cee75a2',
    'SmolLM2-360M': 'f8027fd0eaeea54caa13c31d31b9fdc459c38b49',
}


def request(url):
    return urllib.request.urlopen(url, timeout=60)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def download(model):
    repo = 'HuggingFaceTB/' + model
    pinned = MODEL_REVISIONS[model]
    with request('https://huggingface.co/api/models/' + repo + '/revision/' + pinned + '?blobs=true') as r:
        info = json.load(r)
    revision = info['sha']
    assert revision == pinned
    dest = DEST / model / revision
    dest.mkdir(parents=True, exist_ok=True)
    record = {'repo': repo, 'revision': revision,
              'license': info['cardData'].get('license'), 'files': {}}
    for entry in info['siblings']:
        name = entry['rfilename']
        if name == '.gitattributes':
            continue
        path = dest / name
        expected = entry.get('lfs', {}).get('sha256')
        if not path.exists() or (expected and sha(path) != expected):
            for attempt in range(3):
                try:
                    with request(f'https://huggingface.co/{repo}/resolve/{revision}/{name}') as r, path.with_suffix(path.suffix + '.partial').open('wb') as out:
                        while b := r.read(8 * 1024 * 1024):
                            out.write(b)
                    path.with_suffix(path.suffix + '.partial').replace(path)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2)
        digest = sha(path)
        if expected and digest != expected:
            raise RuntimeError(f'Hash mismatch: {repo}/{name}')
        record['files'][name] = {'sha256': digest, 'bytes': path.stat().st_size}
        print(repo, name, path.stat().st_size, flush=True)
    return model, record


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        records = dict(pool.map(download, ['SmolLM2-135M', 'SmolLM2-360M']))
    (ART / 'model_download_manifest.json').write_text(json.dumps(records, indent=2) + '\n')
