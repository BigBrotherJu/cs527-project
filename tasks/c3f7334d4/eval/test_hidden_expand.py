"""Hidden correctness tests for bgmv_moe_expand (evaluation only, never in the image).

The visible tests stop at 32 tokens and ranks 16/32, so they never reach the
num_pairs >= 256 regime or rank 8/64. These cover every compiled rank, small and
large pair counts, fp16, both finalize modes, several slices, pairs whose LoRA id
is -1, and padded sorted_token_ids entries (== num_tokens), all against an fp32
torch reference.
"""

import pytest
import torch

from flashinfer.fused_moe.bgmv_moe import bgmv_moe_expand, fill_w_ptr


def _inputs(rank, hidden, tokens, top_k, num_slices, dtype, num_experts=16, num_loras=4, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    num_pairs = tokens * top_k
    lora_b = [
        torch.randn(num_loras, num_experts, hidden, rank, dtype=dtype, device=dev, generator=g) * 0.05
        for _ in range(num_slices)
    ]
    w_ptr = torch.zeros(num_slices, num_experts, dtype=torch.int64, device=dev)
    lora_stride = 0
    for s in range(num_slices):
        lora_stride = fill_w_ptr(w_ptr, lora_b[s], num_experts, s)
    x = torch.randn(num_slices, num_pairs, rank, dtype=dtype, device=dev, generator=g)
    sorted_token_ids = torch.randperm(tokens * top_k, device=dev, generator=g) % tokens
    # Padding entries as produced by MoE alignment: token id == num_tokens, to be skipped.
    sorted_token_ids[::7] = tokens
    expert_ids = torch.randint(0, num_experts, (num_pairs,), device=dev, dtype=torch.int64, generator=g)
    topk_weights = torch.rand(num_pairs, device=dev, generator=g)
    lora_indices = torch.randint(0, num_loras, (tokens,), device=dev, dtype=torch.int64, generator=g)
    lora_indices[::5] = -1  # no LoRA for these tokens
    slice_start_loc = torch.arange(num_slices, dtype=torch.int64, device=dev) * hidden
    return x, lora_b, w_ptr, lora_stride, sorted_token_ids, expert_ids, topk_weights, lora_indices, slice_start_loc


def _reference(x, lora_b, sorted_token_ids, expert_ids, topk_weights, lora_indices, finalize):
    num_slices, num_pairs, _ = x.shape
    hidden = lora_b[0].shape[2]
    tokens = lora_indices.numel()
    rows = tokens if finalize else num_pairs
    out = torch.zeros(rows, num_slices * hidden, dtype=torch.float32, device=x.device)
    valid = sorted_token_ids < tokens
    tok = sorted_token_ids.clamp(max=tokens - 1)
    lora = lora_indices[tok]
    valid &= lora >= 0
    pairs = valid.nonzero().squeeze(1)
    for s, w in enumerate(lora_b):
        wsel = w[lora[pairs], expert_ids[pairs]].float()              # [p, hidden, rank]
        contrib = torch.bmm(wsel, x[s, pairs].float().unsqueeze(-1)).squeeze(-1)
        cols = slice(s * hidden, (s + 1) * hidden)
        if finalize:
            out[:, cols].index_add_(0, tok[pairs], contrib * topk_weights[pairs].unsqueeze(-1))
        else:
            out[pairs, cols] = contrib
    return out


@pytest.fixture(autouse=True)
def _require_supported_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor not in {90, 100, 103}:
        pytest.skip("BGMV MoE is validated on SM90/SM100/SM103 only")


@pytest.mark.parametrize("rank", [8, 16, 32, 64])
@pytest.mark.parametrize("hidden", [768, 2688, 4096])
@pytest.mark.parametrize("tokens", [3, 200, 1500])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("finalize", [True, False])
def test_expand_matches_reference(rank, hidden, tokens, dtype, finalize):
    top_k = 2
    num_slices = 2 if hidden == 768 else 1
    (x, lora_b, w_ptr, lora_stride, sorted_token_ids, expert_ids, topk_weights,
     lora_indices, slice_start_loc) = _inputs(rank, hidden, tokens, top_k, num_slices, dtype)
    rows = tokens if finalize else tokens * top_k
    y = torch.zeros(rows, num_slices * hidden, dtype=torch.float32, device="cuda")
    bgmv_moe_expand(y, x, w_ptr, sorted_token_ids, expert_ids, topk_weights, lora_indices,
                    slice_start_loc, [hidden] * num_slices, lora_stride, finalize=finalize)
    ref = _reference(x, lora_b, sorted_token_ids, expert_ids, topk_weights, lora_indices, finalize)
    torch.testing.assert_close(y, ref, rtol=2e-3, atol=2e-3)
