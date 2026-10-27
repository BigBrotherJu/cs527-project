"""Time flashinfer.fused_moe.bgmv_moe_expand alone on a fixed set of shapes.

Only the expand call is inside the timed region: inputs (including the shrink
output it consumes) are built once per shape. GPU time comes from
flashinfer.testing.bench_gpu_time (CUPTI when available, else CUDA events) with
a cold L2 before every iteration; the median per shape is reported.

    python /task/bench_task.py                      # shapes in /task/shapes.json
    python /task/bench_task.py --shapes other.json  # e.g. hidden shapes at evaluation
    python /task/bench_task.py --check              # also compare against a torch reference
    python /task/bench_task.py --json out.json      # machine-readable results

expand accumulates into y with atomicAdd, so repeated timed calls keep adding
to the same buffer; that does not change the work done per call. --check runs
on a freshly zeroed y.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch

from flashinfer.fused_moe.bgmv_moe import bgmv_moe_expand, fill_w_ptr
from flashinfer.testing import bench_gpu_time

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def make_inputs(rank, hidden, tokens, num_experts, top_k, num_loras, num_slices, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    num_pairs = tokens * top_k
    lora_b = [
        torch.randn(num_loras, num_experts, hidden, rank, dtype=dtype, device=dev, generator=g) * 0.01
        for _ in range(num_slices)
    ]
    w_ptr = torch.zeros(num_slices, num_experts, dtype=torch.int64, device=dev)
    lora_stride = 0
    for s in range(num_slices):
        lora_stride = fill_w_ptr(w_ptr, lora_b[s], num_experts, s)
    x = torch.randn(num_slices, num_pairs, rank, dtype=dtype, device=dev, generator=g) * 0.1
    sorted_token_ids = torch.arange(tokens, device=dev, dtype=torch.int64).repeat_interleave(top_k)
    expert_ids = torch.randint(0, num_experts, (num_pairs,), device=dev, dtype=torch.int64, generator=g)
    topk_weights = torch.softmax(
        torch.randn(tokens, top_k, device=dev, generator=g), dim=-1
    ).reshape(-1).float()
    lora_indices = torch.randint(0, num_loras, (tokens,), device=dev, dtype=torch.int64, generator=g)
    slice_start_loc = torch.arange(num_slices, dtype=torch.int64, device=dev) * hidden
    y = torch.zeros(tokens, num_slices * hidden, dtype=torch.float32, device=dev)
    args = (y, x, w_ptr, sorted_token_ids, expert_ids, topk_weights, lora_indices,
            slice_start_loc, [hidden] * num_slices, lora_stride)
    return args, lora_b


def reference(args, lora_b, chunk=256):
    """y[token, s*hidden:(s+1)*hidden] += w * (lora_b[lora, expert] @ x[s, pair]), in fp32."""
    (_, x, _, sorted_token_ids, expert_ids, topk_weights, lora_indices, _, slices, _) = args
    hidden = slices[0]
    tokens = lora_indices.numel()
    out = torch.zeros(tokens, len(slices) * hidden, dtype=torch.float32, device=x.device)
    lora_ids = lora_indices[sorted_token_ids]
    for s, w in enumerate(lora_b):
        for p0 in range(0, x.shape[1], chunk):
            p = slice(p0, p0 + chunk)
            wsel = w[lora_ids[p], expert_ids[p]].float()              # [c, hidden, rank]
            contrib = torch.bmm(wsel, x[s, p].float().unsqueeze(-1)).squeeze(-1)
            contrib *= topk_weights[p].unsqueeze(-1)
            out[:, s * hidden:(s + 1) * hidden].index_add_(0, sorted_token_ids[p], contrib)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=str(Path(__file__).with_name("shapes.json")))
    ap.add_argument("--check", action="store_true", help="compare against a torch reference")
    ap.add_argument("--json", help="write results to this file")
    ap.add_argument("--repeat-ms", type=int, default=200)
    a = ap.parse_args()

    spec = json.loads(Path(a.shapes).read_text())
    results = []
    print(f"{'shape':<22}{'median us':>12}{'min us':>10}{'iters':>7}" + ("  check" if a.check else ""))
    for sh in spec["shapes"]:
        cfg = {**spec.get("defaults", {}), **sh}
        args, lora_b = make_inputs(
            cfg["rank"], cfg["hidden"], cfg["tokens"], cfg["num_experts"], cfg["top_k"],
            cfg["num_loras"], cfg["num_slices"], DTYPES[cfg["dtype"]],
        )
        status = ""
        if a.check:
            args[0].zero_()
            bgmv_moe_expand(*args)
            torch.cuda.synchronize()
            ok = torch.allclose(args[0], reference(args, lora_b), rtol=1e-3, atol=1e-4)
            status = "  ok" if ok else "  MISMATCH"
        times_ms = bench_gpu_time(
            bgmv_moe_expand, input_args=args, enable_cupti=True,
            cold_l2_cache=True, repeat_time_ms=a.repeat_ms,
        )
        med = statistics.median(times_ms) * 1e3
        results.append({"name": cfg["name"], "median_us": med, "min_us": min(times_ms) * 1e3,
                        "iters": len(times_ms), **({"check": status.strip()} if a.check else {})})
        print(f"{cfg['name']:<22}{med:>12.2f}{min(times_ms) * 1e3:>10.2f}{len(times_ms):>7}{status}")
        del args, lora_b
        torch.cuda.empty_cache()

    print(f"device: {torch.cuda.get_device_name()}")
    if a.json:
        Path(a.json).write_text(json.dumps({"device": torch.cuda.get_device_name(), "results": results}, indent=1))


if __name__ == "__main__":
    main()
