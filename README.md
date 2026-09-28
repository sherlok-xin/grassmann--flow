# Grassmann Flows

An independent reproduction study of "Attention Is Not What You Need" (arXiv 2512.19428).

For the current Grassmann--Transformer distillation project, start with [PROJECT_OVERVIEW_FOR_GPT.md](PROJECT_OVERVIEW_FOR_GPT.md). It provides the active code paths, newest controlled results, evidence boundaries, and a reading order that does not require local checkpoints or chat history.

## Summary

This repository contains a reproduction of Grassmann flow layers for sequence modeling. The original paper claims performance "within 10-15% of size-matched Transformers" on Wikitext-2. Our reproduction shows a **22.6% gap** - significantly larger than claimed.

## Current Research Status

The repository now also studies knowledge distillation from a heterogeneous Grassmann--Transformer late-fusion teacher into smaller students. After the CAC submission was rejected, the project returned to empirical validation. Token-normalized matched controls are complete on PTB, and a three-seed WikiText-2 intervention confirms that teacher condition changes KD gain. The bounded branch-aware prototype did not pass its gate and is retained as a negative result.

Start with [the project map](docs/project_map.md), [the evidence handoff](AI_RESEARCH_HANDOFF.md), and [the Phase 2C report](research/experiments/phase2c_multiseed_teacher_utility/REPORT.md). The generated registry at `docs/generated/experiment_registry.csv` indexes historical run directories without moving or deleting their local artifacts.

## Historical Reproduction Result

| Model | Parameters | Test PPL |
|-------|------------|----------|
| Grassmann (paper arch) | 17.70M | 242.94 |
| Transformer | 17.67M | 198.17 |

**Gap: 22.6%** (vs claimed 10-15%)

## Current Controlled KD Result

On WikiText-2, the paired test-NLL advantage of the alpha-0.5 teacher over the alpha-0.0 teacher is positive for all three independent student seeds: 0.115731, 0.113739, and 0.114553. The mean is 0.114674 with sample standard deviation 0.001001. The globally weaker alpha-0.0 teacher still provides a smaller positive KD gain for all seeds, so global teacher NLL does not determine transfer sign.

## CUDA Optimization

Custom CUDA kernels provide **2x inference speedup**:

| Metric | PyTorch | CUDA | Speedup |
|--------|---------|------|---------|
| Full model inference | 9.16 ms | 4.53 ms | 2.0x |

## Blog Post

Full analysis and discussion: [blog.md](blog.md)

## Quick Start

```bash
# Install dependencies
pip install torch datasets transformers tqdm

# Run reproduction
python train_wikitext2.py --model both --epochs 20

# Build CUDA kernels (optional)
cd src/cuda && python setup.py install
```

## Files

- `train_wikitext2.py` - Training script
- `src/models/grassmann_v4.py` - Paper-exact implementation
- `src/cuda/` - CUDA kernel implementation
- `blog.md` - Full reproduction report
- `technical.md` - Technical details

## Hardware

Experiments run on NVIDIA H100 SXM5 80GB (Voltage Park Cloud).

## Citation

```bibtex
@article{arledge2025grassmann,
  title={Grassmann Flows for Sequence Modeling: An Independent Reproduction Study},
  author={Arledge, Elliot},
  year={2025},
  month={December},
  url={https://github.com/Infatoshi/grassmann-flows}
}
```

## License

MIT
