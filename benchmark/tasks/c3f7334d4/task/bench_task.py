"""Time flashinfer.fused_moe.bgmv_moe_expand alone on a fixed set of shapes.

    python /task/bench_task.py                      # shapes in /task/shapes.json
    python /task/bench_task.py --shapes other.json  # e.g. hidden shapes at evaluation
    python /task/bench_task.py --check              # also compare against a torch reference
    python /task/bench_task.py --json out.json      # machine-readable results

Only the expand call is timed; its inputs, including the shrink output it
consumes, are built once per shape. Timing uses CUDA events with a cold L2
before every iteration, and the median per shape is reported. CUDA events
rather than CUPTI, because the image has no cupti-python and every arm of a
comparison must use the same timer.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch

from flashinfer.fused_moe.bgmv_moe import bgmv_moe_expand, fill_w_ptr
from flashinfer.testing import bench_gpu_time

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def make_inputs(cfg, seed=0):
    """Random inputs for one shape, in bgmv_moe_expand's argument order.

    Each token is routed to top_k random experts and uses one random LoRA
    adapter. Returns (args, lora_b), where lora_b is the list of per-slice
    weight banks that the reference reads directly.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    rnd = dict(device="cuda", generator=g)
    rank, hidden, tokens = cfg["rank"], cfg["hidden"], cfg["tokens"]
    experts, top_k, loras, slices = cfg["num_experts"], cfg["top_k"], cfg["num_loras"], cfg["num_slices"]
    dtype = DTYPES[cfg["dtype"]]
    pairs = tokens * top_k

    # LoRA-B weights, [loras, experts, hidden, rank] per slice. The kernel reaches
    # them through a [slices, experts] table of base pointers plus a stride.
    lora_b = [torch.randn(loras, experts, hidden, rank, dtype=dtype, **rnd) * 0.01 for _ in range(slices)]
    w_ptr = torch.zeros(slices, experts, dtype=torch.int64, device="cuda")
    for s in range(slices):
        lora_stride = fill_w_ptr(w_ptr, lora_b[s], experts, s)

    # Shrink output (the expand input) and routing: pair p belongs to token
    # p // top_k, goes to expert_ids[p] and is weighted by topk_weights[p].
    x = torch.randn(slices, pairs, rank, dtype=dtype, **rnd) * 0.1
    sorted_token_ids = torch.arange(tokens, device="cuda").repeat_interleave(top_k)
    expert_ids = torch.randint(0, experts, (pairs,), dtype=torch.int64, **rnd)
    topk_weights = torch.softmax(torch.randn(tokens, top_k, **rnd), dim=-1).reshape(-1)
    lora_indices = torch.randint(0, loras, (tokens,), dtype=torch.int64, **rnd)

    # Output: fp32, slices side by side along the hidden dimension.
    slice_start_loc = torch.arange(slices, dtype=torch.int64, device="cuda") * hidden
    y = torch.zeros(tokens, slices * hidden, dtype=torch.float32, device="cuda")

    args = (y, x, w_ptr, sorted_token_ids, expert_ids, topk_weights, lora_indices,
            slice_start_loc, [hidden] * slices, lora_stride)
    return args, lora_b


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------

def reference(args, lora_b, chunk=256):
    """y[token, slice s] += topk_weight * (lora_b[s][lora, expert] @ x[s, pair]), in fp32.

    Pairs are processed in chunks to bound the memory of the gathered weights.
    """
    _, x, _, sorted_token_ids, expert_ids, topk_weights, lora_indices, _, slice_sizes, _ = args
    hidden = slice_sizes[0]
    out = torch.zeros(lora_indices.numel(), len(slice_sizes) * hidden, dtype=torch.float32, device="cuda")
    lora_ids = lora_indices[sorted_token_ids]
    for s, w in enumerate(lora_b):
        for p0 in range(0, x.shape[1], chunk):
            p = slice(p0, p0 + chunk)
            w_sel = w[lora_ids[p], expert_ids[p]].float()                     # [chunk, hidden, rank]
            contrib = torch.bmm(w_sel, x[s, p].float().unsqueeze(-1)).squeeze(-1)  # [chunk, hidden]
            contrib *= topk_weights[p].unsqueeze(-1)
            out[:, s * hidden:(s + 1) * hidden].index_add_(0, sorted_token_ids[p], contrib)
    return out


def check(args, lora_b):
    """Run expand once on a zeroed output and compare it with the reference."""
    y = args[0]
    y.zero_()
    bgmv_moe_expand(*args)
    torch.cuda.synchronize()
    return "ok" if torch.allclose(y, reference(args, lora_b), rtol=1e-3, atol=1e-4) else "MISMATCH"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=str(Path(__file__).with_name("shapes.json")))
    ap.add_argument("--check", action="store_true", help="compare against a torch reference")
    ap.add_argument("--json", help="write results to this file")
    ap.add_argument("--repeat-ms", type=int, default=200)
    a = ap.parse_args()

    spec = json.loads(Path(a.shapes).read_text())
    device = torch.cuda.get_device_name()
    print(f"{'shape':<26}{'median us':>12}{'min us':>10}{'iters':>7}" + ("  check" if a.check else ""))

    results = []
    for shape in spec["shapes"]:
        cfg = {**spec.get("defaults", {}), **shape}
        args, lora_b = make_inputs(cfg)
        row = {"name": cfg["name"]}
        if a.check:
            row["check"] = check(args, lora_b)

        # expand accumulates into y, so the timed calls keep adding to the same
        # buffer; the work per call is unchanged.
        times_ms = bench_gpu_time(bgmv_moe_expand, input_args=args, enable_cupti=False,
                                  cold_l2_cache=True, repeat_time_ms=a.repeat_ms)
        row.update(median_us=statistics.median(times_ms) * 1e3, min_us=min(times_ms) * 1e3,
                   iters=len(times_ms))
        results.append(row)
        print(f"{row['name']:<26}{row['median_us']:>12.2f}{row['min_us']:>10.2f}{row['iters']:>7}"
              + (f"  {row['check']}" if a.check else ""))

        # Free this shape's weights before building the next one.
        del args, lora_b
        torch.cuda.empty_cache()

    print(f"device: {device}")
    if a.json:
        Path(a.json).write_text(json.dumps({"device": device, "results": results}, indent=1))
    if any(r.get("check") == "MISMATCH" for r in results):
        raise SystemExit("check failed: output differs from the torch reference")


if __name__ == "__main__":
    main()
