"""Immutable compact token streams. Existing dataset directories are read-only."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

ART = Path(__file__).resolve().parents[1]
ROOT = ART.parents[2]
OUT = ROOT / 'outputs/phase4a_modern/data'
COUNTS = {'train': 20_000_000, 'validation': 1_000_000, 'test': 1_000_000}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


class Writer:
    def __init__(self, name):
        self.path = OUT / name
        self.path.mkdir(parents=True, exist_ok=True)
        if (self.path / 'manifest.json').exists():
            raise RuntimeError('Already materialized; verify and reuse immutable files')
        self.files = {s: (self.path / (s + '.bin')).open('xb') for s in COUNTS}
        self.order = gzip.open(self.path / 'document_order.jsonl.gz', 'wt')
        self.counts = {s: 0 for s in COUNTS}
        self.docs = {s: 0 for s in COUNTS}
        self.seen = set()

    def add(self, split, docid, text, ids):
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in self.seen:
            return
        self.seen.add(digest)
        remaining = COUNTS[split] - self.counts[split]
        if remaining <= 0:
            return
        used = ids[:remaining]
        if not used:
            return
        self.files[split].write(np.asarray(used, dtype='<u4').tobytes())
        self.order.write(json.dumps({'split': split, 'id': str(docid), 'text_sha256': digest,
                                     'tokens_used': len(used), 'tokens_full': len(ids)}) + '\n')
        self.counts[split] += len(used)
        self.docs[split] += 1

    def finish(self, source):
        for f in self.files.values():
            f.close()
        self.order.close()
        assert self.counts == COUNTS, self.counts
        result = {'source': source, 'token_counts': self.counts, 'documents': self.docs,
                  'packing': 'concatenate documents with one EOS; no BOS; no added tokens',
                  'token_dtype': 'little-endian uint32', 'sha256': {s: sha(self.path / (s + '.bin')) for s in COUNTS},
                  'document_order_sha256': sha(self.path / 'document_order.jsonl.gz')}
        (self.path / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
        (ART / ('data_manifest_' + self.path.name + '.json')).write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2), flush=True)


def main(name):
    models = json.loads((ART / 'model_download_manifest.json').read_text())
    path = ROOT / 'outputs/phase4a_modern/model_cache/SmolLM2-135M' / models['SmolLM2-135M']['revision']
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    tokenizer.model_max_length = 10 ** 12
    writer = Writer(name)
    if name == 'tinystories':
        source_path = ROOT.parent / 'datasets/tinystories_saved'
        all_data = load_from_disk(str(source_path))
        split = all_data['train'].train_test_split(test_size=0.02, seed=42, shuffle=True,
                     train_indices_cache_file_name=str(writer.path / 'train_indices.arrow'),
                     test_indices_cache_file_name=str(writer.path / 'validation_indices.arrow'),
                     load_from_cache_file=False)
        sources = {'train': split['train'], 'validation': split['test'], 'test': all_data['validation']}
        for s, ds in sources.items():
            for start in range(0, len(ds), 256):
                texts = ds[start:start + 256]['text']
                enc = tokenizer(texts, add_special_tokens=False)['input_ids']
                for j, (text, ids) in enumerate(zip(texts, enc)):
                    row = start + j
                    original = row if ds._indices is None else int(ds._indices.column(0)[row].as_py())
                    writer.add(s, ('validation' if s == 'test' else 'train') + ':' + str(original), text, ids + [tokenizer.eos_token_id])
                if writer.counts[s] >= COUNTS[s]:
                    break
                if start % (256 * 20) == 0:
                    print(s, writer.counts[s], flush=True)
        source = {'repo': 'roneneldan/TinyStories', 'historical_hf_revision': 'UNKNOWN_LOCAL_SAVED_SNAPSHOT',
                  'local_path': str(source_path), 'fingerprints': {s: ds._fingerprint for s, ds in all_data.items()},
                  'source_file_sha256': {str(p.relative_to(source_path)): sha(p) for p in sorted(source_path.rglob('*')) if p.is_file() and (p.name.startswith('data-') or p.suffix == '.json')},
                  'split': 'original train.train_test_split(test_size=0.02, seed=42, shuffle=True); original validation is final test',
                  'indices_cache_location': 'new output data namespace only', 'deduplicate_text': True}
    else:
        meta = json.loads((ART / 'fineweb_source_metadata.json').read_text())
        revision = meta['sha']
        # The mirror returns pagination links to the inaccessible main domain.
        # Read the same pinned config's first shard directly, without repo listing.
        source_url = f'https://hf-mirror.com/datasets/HuggingFaceFW/fineweb-edu/resolve/{revision}/sample/10BT/000_00000.parquet'
        ds = load_dataset('parquet', data_files={'train': [source_url]}, split='train', streaming=True,
                          cache_dir=str(OUT / 'hf_metadata_cache'))
        for i, batch in enumerate(ds.iter(batch_size=64)):
            enc = tokenizer(batch['text'], add_special_tokens=False)['input_ids']
            for j, (text, ids) in enumerate(zip(batch['text'], enc)):
                docid = str(batch.get('id', batch.get('url', []))[j])
                value = int(hashlib.sha256(docid.encode()).hexdigest(), 16) % 22
                s = 'validation' if value == 20 else 'test' if value == 21 else 'train'
                writer.add(s, docid, text, ids + [tokenizer.eos_token_id])
            if i % 20 == 0:
                print(writer.counts, flush=True)
            if writer.counts == COUNTS:
                break
        source = {'repo': 'HuggingFaceFW/fineweb-edu', 'revision': revision, 'config': 'sample-10BT',
                  'source_shards': ['sample/10BT/000_00000.parquet'],
                  'origin_url': source_url.replace('hf-mirror.com', 'huggingface.co'),
                  'transport': 'direct pinned parquet streaming through hf-mirror; bypass unavailable listing pagination',
                  'order': 'pinned official streaming shard/row order, no shuffle',
                  'split': 'SHA256(document ID) modulo 22: 0..19 train, 20 validation, 21 test',
                  'deduplicate_text': True, 'pretraining_overlap': 'FineWeb-Edu included in SmolLM2 pretraining; exact overlap unknown'}
    writer.finish(source)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('dataset', choices=['tinystories', 'fineweb'])
    main(p.parse_args().dataset)
