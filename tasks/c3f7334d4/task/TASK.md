# Task: speed up the MoE BGMV expand kernel

The repository at `/workspace/flashinfer` is FlashInfer, a CUDA kernel library
for LLM serving. It is installed in editable mode and kernels are JIT-compiled:
edit the sources and rerun, no reinstall needed.

## Hotspot

`flashinfer.fused_moe.bgmv_moe_expand`, the expand step of multi-LoRA MoE
(`y[token] += topk_weight * (x[slice, pair] @ lora_b[expert, lora])`). The
CUDA kernels and their dispatch are in `csrc/bgmv_moe/moe_bgmv_impl.cuh`.

## Diagnosis from the performance engineer

The expand op is memory-bound: every (token, expert) pair reads a
`feat_out x feat_in` slice of LoRA-B. The reads of the weight matrix W are
poorly coalesced: profiling measured about 7.6 L2 sectors per global load
request on W, against about 4 for a fully coalesced access, so roughly half of
each memory transaction is wasted. The gap grows with the number of pairs.

## What to do

Make `bgmv_moe_expand` faster on the benchmark shapes without regressing the
other shapes and dtypes it supports.

- Keep the results correct: fp32 accumulation into `y`, the same semantics for
  both `finalize` modes, skipped pairs (`lora_id < 0`) and every compiled
  rank / hidden size.
- Do not modify `tests/`, `benchmarks/` or `/task`; changes there are discarded
  at evaluation.
- Your solution is evaluated on the visible tests and benchmark shapes and on
  hidden ones (other ranks, hidden sizes, token counts and fp16).

## Tools

    pytest -q tests/moe/test_bgmv_moe.py tests/moe/test_bgmv_moe_lora_delta.py
    python /task/bench_task.py            # median GPU time per shape (us)
    python /task/bench_task.py --check    # plus a check against a torch reference

The shapes timed are in `/task/shapes.json`. There is no network access.
