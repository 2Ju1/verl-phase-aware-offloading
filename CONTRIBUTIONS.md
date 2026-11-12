# Contribution map

This repository separates the upstream framework, the modified implementation,
and experiment evidence so that each claim can be traced to code and data.

## Upstream boundary

- Upstream project: [`verl-project/verl`](https://github.com/verl-project/verl)
- Modified fork: [`2Ju1/verl`](https://github.com/2Ju1/verl), branch
  `single-gpu-phase-offloading`
- Pinned implementation commit: `8fb2e22709bb6fe5ec22009148901de1cae5370a`
- Baseline described by the study: v0.4.1-derived

The framework is not copied into this repository. `src/verl` is a Git
submodule, preserving upstream history and keeping the implementation boundary
visible.

## Change history

The pinned implementation is built as three reviewable commits on top of the
v0.4.1-derived baseline:

| Commit | Scope |
|---|---|
| `520c1c95` | Implement single-GPU phase-aware offloading |
| `b91cd475` | Add tests for phase-aware offloading |
| `8fb2e227` | Add benchmark drivers and experiment documentation |

Compared with baseline commit `8d9e350e`, the change covers 68 files with
8,172 additions and 66 deletions. Targeted verification lives in:

```text
src/verl/tests/workers/actor/test_single_gpu_offload.py
src/verl/tests/utils/test_activation_offload.py
src/verl/benchmarks/offload/test_tools.py
```

## Implemented system

The work adds three cooperating layers to single-GPU full-parameter GRPO:

1. **Phase-aware residency** moves inactive Actor and Reference parameters out
   of GPU memory at phase boundaries.
2. **CPU AdamW** keeps FP32 master parameters, gradients, and optimizer state
   on CPU, removing the optimizer-state GPU peak.
3. **Bounded gradient streaming** transfers ready gradients through reusable
   buckets, releases GPU gradients early, and pipelines CPU Adam updates with
   updated-parameter H2D copies.

Primary implementation entry points inside the submodule:

```text
src/verl/verl/workers/actor/single_gpu_offload/engine.py
src/verl/verl/workers/actor/single_gpu_offload/streaming.py
src/verl/verl/workers/actor/single_gpu_offload/pipeline.py
src/verl/benchmarks/offload/run_matrix.py
```

## Evidence map

| Claim | Evidence |
|---|---|
| Phase offload lowers Actor-backward peak | `results/data/02_allgpu_vs_phase_offload/` |
| CPU AdamW removes the Update GPU peak | `results/data/01_phase_offload_vs_cpu_adamw/` |
| 16 MiB streaming lowers backward peak and step time | `results/data/03_nostream_vs_stream16/` |
| Streaming enables the 1.5B capacity result | `results/data/04_qwen15b_capacity/` |
| Historical bucket-size diagnostic | `results/data/05_bucket_size_sweep_05b/` |

Each run directory retains its configuration and result JSON. Failed OOM runs
are kept as capacity evidence and are excluded from performance means.

## Review path

1. Read the top-level `README.md` for the question and quantitative results.
2. Inspect the implementation entry points above in the pinned submodule.
3. Read `docs/ENGINEERING_INSIGHTS_KO.md` for debugging decisions and rejected
   interpretations.
4. Use `docs/EXPERIMENTS.md` and the evidence map to reproduce or audit a
   comparison.

