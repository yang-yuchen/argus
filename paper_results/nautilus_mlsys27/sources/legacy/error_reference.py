import math
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import neptune
import numpy as np
import seaborn as sns
import torch
import tqdm
from neptune.operators import create_general_attention, schedule_attn_flash
from tvm import runtime
from tvm.target.target import Target
from tvm.tir import Schedule

SEQ_LEN = 2048
NUM_RUNS = 1000
CACHE_PATH = Path("~/.neptune/real_qkv_cache.pt").expanduser()
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
NEPTUNE_RUNNABLE = None


def naive_attention(q, k, v):
    d = q.shape[-1]
    qk = q @ k.transpose(-2, -1)
    qk = qk / math.sqrt(d)
    qk = torch.softmax(qk, dim=-1)
    return qk @ v


def torch_attention(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


def flash_attn(q, k, v) -> torch.Tensor:
    import flash_attn

    q, k, v = [x.transpose(1, 2).contiguous() for x in [q, k, v]]
    out = cast(torch.Tensor, flash_attn.flash_attn_func(q, k, v))
    return out.transpose(1, 2)


def neptune_attention(q, k, v):
    assert q.dtype == k.dtype == v.dtype == torch.float16
    global NEPTUNE_RUNNABLE
    if NEPTUNE_RUNNABLE is None:
        b, n, q_len, h = q.shape
        _, _, k_len, _ = k.shape
        attn_mod = create_general_attention((b, n, q_len, k_len, h))
        sch = Schedule(attn_mod)
        schedule_attn_flash(sch, "triton")
        NEPTUNE_RUNNABLE = neptune.compile(sch.mod, Target("cuda").with_host("llvm"))
    device = runtime.device("cuda", 0)
    nq, nk, nv = [runtime.ndarray.array(data.cpu(), device=device) for data in [q, k, v]]
    fp16 = torch.float16
    no = runtime.ndarray.array(torch.empty(q.shape, dtype=fp16, device="cpu"), device=device)
    NEPTUNE_RUNNABLE(nq, nk, nv, no)
    return torch.from_numpy(no.numpy())


def run_attn_get_diffs(tq, tk, tv):
    fp16, fp64 = torch.float16, torch.float64
    assert tq.dtype == tk.dtype == tv.dtype == fp64
    assert tq.device == tk.device == tv.device
    naive_output = naive_attention(tq, tk, tv).cpu()
    tq, tk, tv = tq.to(fp16), tk.to(fp16), tv.to(fp16)
    torch_output = torch_attention(tq, tk, tv).cpu().to(fp64)
    neptune_output = neptune_attention(tq, tk, tv).cpu().to(fp64)
    fa_output = flash_attn(tq, tk, tv).cpu().to(fp64)
    torch_abs_error = (torch_output - naive_output).flatten().abs()
    neptune_abs_error = (neptune_output - naive_output).flatten().abs()
    fa_abs_error = (fa_output - naive_output).flatten().abs()
    return torch_abs_error, neptune_abs_error, fa_abs_error


@torch.inference_mode()
def _collect_real_qkv_samples(
    num_samples: int, seq_len: int, model_id: str, dataset_name: str, dataset_config: str, seed: int
):
    def _reshape_heads(x, n_heads):
        bsz, seq_len, width = x.shape
        head_dim = width // n_heads
        return x.view(bsz, seq_len, n_heads, head_dim).permute(0, 2, 1, 3).contiguous()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rng = np.random.default_rng(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="cuda"
    ).eval()

    # Runtime optimization: use only the first attention layer projection path.
    block = model.model.layers[0]
    attn = block.self_attn
    config = attn.config
    q_heads = config.num_attention_heads
    kv_heads = config.num_key_value_heads
    if q_heads % kv_heads != 0:
        raise RuntimeError(f"Unsupported heads layout: q_heads={q_heads}, kv_heads={kv_heads}")
    kv_repeat = q_heads // kv_heads

    dataset = load_dataset(dataset_name, dataset_config, split="train", streaming=True)
    samples = []
    joined_text = ""
    pbar = tqdm.tqdm(dataset, desc="Collecting samples")
    for row in pbar:
        pbar.set_postfix({"samples": f"{len(samples)}/{num_samples}"})
        joined_text += "\n" + row.get("text", "").strip()
        tokenized = tokenizer(joined_text, return_tensors="pt", add_special_tokens=False)
        input_ids = tokenized.input_ids[0]
        if input_ids.numel() < seq_len:
            continue
        joined_text = ""
        start = int(rng.integers(0, input_ids.numel() - seq_len + 1))
        chunk = input_ids[start : start + seq_len].unsqueeze(0).to("cuda")

        hidden = model.model.embed_tokens(chunk)
        hidden = block.input_layernorm(hidden)
        q = _reshape_heads(attn.q_proj(hidden), q_heads)
        k = _reshape_heads(attn.k_proj(hidden), kv_heads)
        v = _reshape_heads(attn.v_proj(hidden), kv_heads)
        if kv_repeat > 1:
            k = k.repeat_interleave(kv_repeat, dim=1)
            v = v.repeat_interleave(kv_repeat, dim=1)
        samples.append((q[0].cpu(), k[0].cpu(), v[0].cpu()))
        if len(samples) >= num_samples:
            break
    if len(samples) < num_samples:
        raise RuntimeError(f"Only collected {len(samples)} samples, expected {num_samples}.")
    return samples


def get_real_qkv_samples(
    num_samples=NUM_RUNS,
    seq_len=SEQ_LEN,
    cache_path=CACHE_PATH,
    model_id=MODEL_ID,
    dataset_name=DATASET_NAME,
    dataset_config=DATASET_CONFIG,
    seed=0,
):
    if cache_path.exists():
        print(f"Loading cached samples from {cache_path}")
        payload = torch.load(cache_path, map_location="cpu")
        if payload["seq_len"] == seq_len and len(payload["samples"]) >= num_samples:
            return payload["samples"][:num_samples]

    samples = _collect_real_qkv_samples(
        num_samples=num_samples,
        seq_len=seq_len,
        model_id=model_id,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        seed=seed,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "seq_len": seq_len,
            "model_id": model_id,
            "dataset": f"{dataset_name}/{dataset_config}",
            "samples": samples,
        },
        cache_path,
    )
    return samples


def error_metric(diff):
    return diff.pow(2).mean().sqrt().item()


def capped_bin_edges(*arrays, max_bins=200):
    combined = np.concatenate([np.asarray(arr) for arr in arrays])
    edges = np.histogram_bin_edges(combined, bins="auto")
    n_bins = len(edges) - 1
    if n_bins <= max_bins:
        return edges
    return np.linspace(combined.min(), combined.max(), max_bins + 1)


def plot_single_run_error(torch_abs_error, neptune_abs_error, fa_abs_error):
    fig, ax = plt.subplots(figsize=(8, 5))
    kwargs = {
        "stat": "count",
        "element": "step",
        "bins": capped_bin_edges(torch_abs_error, neptune_abs_error, fa_abs_error),
        "fill": False,
        "ax": ax,
    }
    sns.histplot(torch_abs_error, label="|torch_output - naive_output|", **kwargs)
    sns.histplot(neptune_abs_error, label="|neptune_output - naive_output|", **kwargs)
    sns.histplot(fa_abs_error, label="|flash_attn_output - naive_output|", **kwargs)
    ax.set_title("Absolute Error Histogram")
    ax.set_xlabel("Absolute error")
    ax.set_ylabel("Frequency")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    return fig


def main():
    samples = get_real_qkv_samples(num_samples=NUM_RUNS, seq_len=SEQ_LEN)
    tq, tk, tv = [x.unsqueeze(0).cuda().to(torch.float64) for x in samples[0]]
    torch_diff, neptune_diff, fa_diff = run_attn_get_diffs(tq, tk, tv)
    fig = plot_single_run_error(torch_diff, neptune_diff, fa_diff)
    fig.savefig("absolute_error.pdf")
    plt.close(fig)

    torch_avg_errors = []
    neptune_avg_errors = []
    fa_avg_errors = []
    for q, k, v in tqdm.tqdm(samples, desc="Running tests"):
        tq, tk, tv = [x.unsqueeze(0).cuda().to(torch.float64) for x in [q, k, v]]
        torch_diff, neptune_diff, fa_diff = run_attn_get_diffs(tq, tk, tv)
        torch_avg_errors.append(error_metric(torch_diff))
        neptune_avg_errors.append(error_metric(neptune_diff))
        fa_avg_errors.append(error_metric(fa_diff))
    # bins = capped_bin_edges(torch_avg_errors, neptune_avg_errors, fa_avg_errors)
    fig, ax = plt.subplots(figsize=(8, 5))
    kwargs = {"ax": ax, "alpha": 0.3, "bw_adjust": 0.5, "fill": True}
    sns.kdeplot(torch_avg_errors, label="PyTorch SDPA", **kwargs)
    sns.kdeplot(neptune_avg_errors, label="Neptune", **kwargs)
    sns.kdeplot(fa_avg_errors, label="Tri-Dao", **kwargs)
    ax.set_title("RMSE Density Plot")
    ax.set_xlabel("Root mean square error")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig("rmse.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()