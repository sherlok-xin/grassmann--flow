# Model and data audit

Official base models only: SmolLM2-135M at
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2` and SmolLM2-360M at
`f8027fd0eaeea54caa13c31d31b9fdc459c38b49`. Both licenses are Apache 2.0.
Downloaded weights match the official API LFS SHA256; every file hash is in
model_download_manifest.json. Actual unique parameter counts are 134,515,008
and 361,821,120. Both are LlamaForCausalLM, use tied word embeddings and have
49,152 vocabulary entries. No instruct checkpoint, quantization or adapter.

Both fast tokenizers have identical complete vocabulary maps, special-token
maps and backend serialization. Backend SHA256:
`2225e8fb36485529d3c826106f8bec359b719a6bcc3f5d6314a834c7be153905`.
BOS/EOS/UNK refer to `<|endoftext|>` (ID 0). No new tokens or chat templates.
Text tokenization disables automatic special tokens and appends one EOS.

Remote hardware: four RTX 3090 24GB GPUs, initially idle. NVIDIA PyTorch
2.6.0a0+df5bbc09d1.nv24.11 supports BF16. Transformers 4.57.6 failed import
because TransformGetItemToIndex is missing from torch._dynamo. An isolated
output-directory installation of Transformers 4.46.3/tokenizers 0.20.3
imports and loads both models without modifying any framework source or
the shared virtualenv. Official model config records transformers_version
4.40.1. Four CE/KL direction, gradient, mask, token-budget and scheduler
unit tests pass in the remote environment.

Stage 0: teacher CE peak allocated 7.468 GiB, KD peak 4.540 GiB; microbatch
4, sequence 256. Warmed throughput is approximately 4,904 / 4,872 target
tokens/s. These are disposable repeated-text smoke measurements, not
generalization or target-corpus optimization results. Fullbatch smoke uses
real TinyStories training chunks and eight accumulated microbatches; see
fullbatch_smoke.json. Formal per-run optimization is audited separately.

TinyStories uses the historical saved corpus at the read-only existing path.
The original 2% seed-42 training holdout and original validation-as-test
mapping are retained. New index caches and tokens are written exclusively
under outputs/phase4a_modern/data. Source Arrow file hashes, fingerprints,
token counts and token hashes are in data_manifest_tinystories.json.
Historical upstream commit is UNKNOWN_LOCAL_SAVED_SNAPSHOT, not guessed.
The modern pilot uses a fixed subset and SmolLM2 tokenizer, so its absolute
NLL values are not comparable with the old GPT2-tokenized experiment.

FineWeb-Edu official sample-10BT revision and ODC-BY license are recorded in
fineweb_source_metadata.json. Source URL:
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu . Token subset and
document-order manifest was produced after clean Stage 1 completion; see
data_manifest_fineweb.json. It contains 20M/1M/1M tokens across
18,926/875/926 documents. The official sample-10BT first-shard prefix is
sufficient; only that shard's needed range data was streamed. Mirror directory
pagination returned inaccessible main-domain links, so direct pinned Parquet
streaming bypassed listing. No full 2.15GB shard was downloaded or saved.
The official first shard reports LFS SHA256
b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871
and size 2,152,819,114 bytes. These identify the origin file; the entire-file
hash is not claimed to have been verified from partial range reads. The
materialized token-stream hashes identify the actual pilot inputs.

Pretraining confounds: both official model cards list FineWeb-Edu among
pretraining sources; exact document overlap is unknown. The 135M and 360M
models also have different pretraining budgets (2T and 4T). Thus the pilot
tests a new pretrained model family and target adaptation setting, not a
guarantee of unseen corpus documents or an isolated parameter-count effect.
Official cards:
https://huggingface.co/HuggingFaceTB/SmolLM2-135M and
https://huggingface.co/HuggingFaceTB/SmolLM2-360M .
