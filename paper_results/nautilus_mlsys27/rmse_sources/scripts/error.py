#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

os.environ.setdefault("XDG_CACHE_HOME", "/tmp/argus_xdg_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/argus_mpl_config")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from argus import operators as ops
from argus.utils import bnsh_expand_for_gqa, ensure_triton_cache_dir

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - optional dependency
    sns = None

logger = logging.getLogger(__name__)

CACHE_VERSION = 2
DEFAULT_SEQ_LEN = 2048
DEFAULT_NUM_RUNS = 128
DEFAULT_CACHE_PATH = Path("~/.argus/real_qkv_cache_v2.pt").expanduser()
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_DATASET_NAME = "wikitext"
DEFAULT_DATASET_CONFIG = "wikitext-103-raw-v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "logs" / "error"
DEFAULT_OPERATOR = "gqa"
DEFAULT_BASELINES = (
    "tilelang",
    "flex",
    "tri-dao:cutlass",
    "torch:flash",
    "torch:cudnn",
    "triton:openai",
    "triton:tri-dao",
)

METHOD_LABELS = {
    "nautilus": "Nautilus",
    "tilelang": "TileLang",
    "flex": "FlexAttn",
    "tri-dao:cutlass": "FlashAttention-2",
    "torch:flash": "PyTorch Flash",
    "torch:cudnn": "cuDNN",
    "triton:openai": "OpenAI Triton",
    "triton:tri-dao": "Tri-Dao Triton",
}

METHOD_COLORS = {
    "Nautilus": "#2ca02c",
    "TileLang": "#1f77b4",
    "FlexAttn": "#7f7f7f",
    "FlashAttention-2": "#ff7f0e",
    "PyTorch Flash": "#8c564b",
    "cuDNN": "#d62728",
    "OpenAI Triton": "#9467bd",
    "Tri-Dao Triton": "#17becf",
}

OPERATOR_ALIASES = {
    "global": "prefill_global",
    "causal": "prefill_causal",
    "gqa": "prefill_gqa",
}
OPERATOR_CHOICES = (
    "auto",
    *OPERATOR_ALIASES.keys(),
    *OPERATOR_ALIASES.values(),
)


class ErrorRunner(Protocol):
    def setup(self, *inputs) -> bool:
        ...

    def run(self) -> None:
        ...

    def get_output(self) -> torch.Tensor:
        ...

    def free_members_memory(self) -> None:
        ...


@dataclass
class MethodSpec:
    key: str
    label: str
    factory: Callable[[], ErrorRunner]


@dataclass
class ActiveMethod:
    key: str
    label: str
    runner: ErrorRunner
    values: list[float]
    enabled: bool = True


@dataclass
class ExperimentSummary:
    operator_name: str
    output_path: Path
    methods: dict[str, list[float]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RMSE density plots for Argus attention kernels")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--num-runs", type=int, default=DEFAULT_NUM_RUNS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the RMSE PDF. Defaults to logs/error/<operator>_rmse.pdf",
    )
    parser.add_argument(
        "--results-output",
        type=Path,
        default=None,
        help="Where to save raw RMSE values and summary statistics as JSON. "
        "Defaults to logs/error/<operator>_results.json",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Where to save the run log. Defaults to logs/error/<operator>.log",
    )
    parser.add_argument(
        "--operator",
        choices=OPERATOR_CHOICES,
        default=DEFAULT_OPERATOR,
        help="Prefill operator to evaluate on cached real QKV samples. "
        "Short aliases global/causal/gqa are supported.",
    )
    parser.add_argument("--baselines", default=",".join(DEFAULT_BASELINES))
    parser.add_argument(
        "--flex-max-autotune",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--reference-chunk-size",
        type=int,
        default=128,
        help="Chunk size for FP64 reference attention along the query sequence dimension",
    )
    parser.add_argument(
        "--plot-percentile-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
        help="Choose x-axis limits from pooled plotted values using percentiles, e.g. 0.5 99.5",
    )
    parser.add_argument(
        "--plot-padding-frac",
        type=float,
        default=0.08,
        help="Fractional padding to add around the selected x-axis span",
    )
    return parser.parse_args(argv)


def normalize_operator_name(operator_arg: str) -> str:
    return OPERATOR_ALIASES.get(operator_arg, operator_arg)


def operator_is_causal(operator_name: str) -> bool:
    return operator_name != "prefill_global"


def resolve_artifact_paths(
    operator_name: str,
    *,
    output: Path | None,
    results_output: Path | None,
    log_file: Path | None,
) -> tuple[Path, Path, Path]:
    output_path = (
        output.expanduser()
        if output is not None
        else DEFAULT_ARTIFACT_DIR / f"{operator_name}_rmse.pdf"
    )
    results_path = (
        results_output.expanduser()
        if results_output is not None
        else DEFAULT_ARTIFACT_DIR / f"{operator_name}_results.json"
    )
    log_path = (
        log_file.expanduser() if log_file is not None else DEFAULT_ARTIFACT_DIR / f"{operator_name}.log"
    )
    return output_path, results_path, log_path


def configure_logging(log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
        force=True,
    )


def _safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _reshape_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    bsz, seq_len, width = x.shape
    head_dim = width // n_heads
    return x.view(bsz, seq_len, n_heads, head_dim).permute(0, 2, 1, 3).contiguous()


@torch.inference_mode()
def _collect_real_qkv_samples(
    num_samples: int,
    seq_len: int,
    model_id: str,
    dataset_name: str,
    dataset_config: str,
    seed: int,
):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rng = np.random.default_rng(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda",
    ).eval()

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError(f"Unsupported model structure for {model_id}: expected model.model.layers")
    if not hasattr(model.model, "embed_tokens"):
        raise RuntimeError(f"Unsupported model structure for {model_id}: expected model.model.embed_tokens")

    block = model.model.layers[0]
    if not hasattr(block, "self_attn"):
        raise RuntimeError(f"Unsupported model structure for {model_id}: first block has no self_attn")
    attn = block.self_attn
    config = attn.config

    q_heads = config.num_attention_heads
    kv_heads = getattr(config, "num_key_value_heads", q_heads)
    if q_heads % kv_heads != 0:
        raise RuntimeError(f"Unsupported heads layout: q_heads={q_heads}, kv_heads={kv_heads}")
    head_dim = attn.q_proj.out_features // q_heads
    is_causal = True

    dataset = load_dataset(dataset_name, dataset_config, split="train", streaming=True)
    samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
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
        if hasattr(block, "input_layernorm"):
            hidden = block.input_layernorm(hidden)
        elif hasattr(block, "ln_1"):
            hidden = block.ln_1(hidden)

        q = _reshape_heads(attn.q_proj(hidden), q_heads)[0].to("cpu", dtype=torch.float16).contiguous()
        k = _reshape_heads(attn.k_proj(hidden), kv_heads)[0].to("cpu", dtype=torch.float16).contiguous()
        v = _reshape_heads(attn.v_proj(hidden), kv_heads)[0].to("cpu", dtype=torch.float16).contiguous()
        samples.append((q, k, v))
        if len(samples) >= num_samples:
            break

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if len(samples) < num_samples:
        raise RuntimeError(f"Only collected {len(samples)} samples, expected {num_samples}.")

    return {
        "version": CACHE_VERSION,
        "seq_len": seq_len,
        "model_id": model_id,
        "dataset": f"{dataset_name}/{dataset_config}",
        "seed": seed,
        "is_causal": is_causal,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "samples": samples,
    }


def get_real_qkv_payload(
    num_samples: int,
    seq_len: int,
    cache_path: Path,
    model_id: str,
    dataset_name: str,
    dataset_config: str,
    seed: int,
):
    expected_dataset = f"{dataset_name}/{dataset_config}"
    if cache_path.exists():
        payload = _safe_torch_load(cache_path)
        if (
            payload.get("version") == CACHE_VERSION
            and payload.get("seq_len") == seq_len
            and payload.get("model_id") == model_id
            and payload.get("dataset") == expected_dataset
            and payload.get("seed") == seed
            and len(payload.get("samples", [])) >= num_samples
        ):
            payload["samples"] = payload["samples"][:num_samples]
            logger.info("Loading cached samples from %s", cache_path)
            return payload
        logger.info("Ignoring stale cache at %s and rebuilding it", cache_path)

    payload = _collect_real_qkv_samples(
        num_samples=num_samples,
        seq_len=seq_len,
        model_id=model_id,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        seed=seed,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


def infer_operator_name(
    operator_arg: str,
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    is_causal: bool,
) -> str:
    if operator_arg != "auto":
        return operator_arg

    q_heads = q.shape[0]
    kv_heads = k.shape[0]
    if is_causal and q_heads != kv_heads:
        return "prefill_gqa"
    if is_causal:
        return "prefill_causal"
    if q_heads == kv_heads:
        return "prefill_global"
    raise ValueError(
        "Unable to infer an operator for a non-causal GQA sample; pass --operator explicitly."
    )


def parse_baseline_tokens(spec: str) -> list[str]:
    tokens = [token.strip() for token in spec.split(",") if token.strip()]
    deduped = list(dict.fromkeys(tokens))
    unknown = [token for token in deduped if token not in METHOD_LABELS or token == "nautilus"]
    if unknown:
        raise ValueError(f"Unknown baseline token(s): {', '.join(unknown)}")
    return deduped


def build_method_specs(
    operator_name: str,
    baseline_tokens: list[str],
    flex_max_autotune: bool,
) -> list[MethodSpec]:
    from argus.flashattn import TriDaoRunner
    from argus.neptune_tilelang_pypi import NeptuneTileLangPrefillRunnerBase
    from argus.pytorch.flex import FlexAttentionRunner
    from argus.pytorch.sdpa import AttentionAlgo, TorchSDPARunner
    from argus.tilelang_pypi import TileLangPrefillRunner
    from argus.triton_runners import OpenAITritonRunner, TriDaoTritonRunner

    operator_map = {
        "prefill_global": ops.PF_GLOBAL,
        "prefill_causal": ops.PF_CAUSAL,
        "prefill_gqa": ops.PF_GQA,
    }
    op = operator_map[operator_name]

    if operator_name == "prefill_global":
        main_factory = lambda: NeptuneTileLangPrefillRunnerBase(
            op,
            is_causal=False,
            enable_gqa=False,
            disable_autotuning=True,
        )
        baseline_builders = {
            "tilelang": lambda: TileLangPrefillRunner(
                op,
                is_causal=False,
                enable_gqa=False,
                disable_autotuning=True,
            ),
            "flex": lambda: FlexAttentionRunner(op, max_autotune=flex_max_autotune),
            "tri-dao:cutlass": lambda: TriDaoRunner(op, causal=False),
            "torch:flash": lambda: TorchSDPARunner(op, AttentionAlgo.FLASH, False, False),
            "torch:cudnn": lambda: TorchSDPARunner(op, AttentionAlgo.CUDNN, False, False),
            "triton:openai": lambda: OpenAITritonRunner(op, False),
            "triton:tri-dao": lambda: TriDaoTritonRunner(op, False),
        }
    elif operator_name == "prefill_causal":
        main_factory = lambda: NeptuneTileLangPrefillRunnerBase(
            op,
            is_causal=True,
            enable_gqa=False,
            disable_autotuning=True,
        )
        baseline_builders = {
            "tilelang": lambda: TileLangPrefillRunner(
                op,
                is_causal=True,
                enable_gqa=False,
                disable_autotuning=True,
            ),
            "flex": lambda: FlexAttentionRunner(
                op,
                mask_expr=ops.causal_mask_cond_4d,
                max_autotune=flex_max_autotune,
            ),
            "tri-dao:cutlass": lambda: TriDaoRunner(op, causal=True),
            "torch:flash": lambda: TorchSDPARunner(op, AttentionAlgo.FLASH, True, False),
            "torch:cudnn": lambda: TorchSDPARunner(op, AttentionAlgo.CUDNN, True, False),
            "triton:openai": lambda: OpenAITritonRunner(op, True),
            "triton:tri-dao": lambda: TriDaoTritonRunner(op, True),
        }
    else:
        main_factory = lambda: NeptuneTileLangPrefillRunnerBase(
            op,
            is_causal=True,
            enable_gqa=True,
            disable_autotuning=True,
        )
        baseline_builders = {
            "tilelang": lambda: TileLangPrefillRunner(
                op,
                is_causal=True,
                enable_gqa=True,
                disable_autotuning=True,
            ),
            "flex": lambda: FlexAttentionRunner(
                op,
                mask_expr=ops.causal_mask_cond_4d,
                enable_gqa=True,
                max_autotune=flex_max_autotune,
            ),
            "tri-dao:cutlass": lambda: TriDaoRunner(op, causal=True),
            "torch:flash": lambda: TorchSDPARunner(op, AttentionAlgo.FLASH, True, True),
            "torch:cudnn": lambda: TorchSDPARunner(op, AttentionAlgo.CUDNN, True, True),
            "triton:openai": lambda: OpenAITritonRunner(op, True),
            "triton:tri-dao": lambda: TriDaoTritonRunner(op, True),
        }

    specs = [MethodSpec("nautilus", METHOD_LABELS["nautilus"], main_factory)]
    specs.extend(
        MethodSpec(token, METHOD_LABELS[token], baseline_builders[token]) for token in baseline_tokens
    )
    return specs


def naive_attention_fp64(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool,
    chunk_size: int,
) -> torch.Tensor:
    q64 = q.to(torch.float64)
    k64 = k.to(torch.float64)
    v64 = v.to(torch.float64)
    if q64.shape[1] != k64.shape[1]:
        q64, k64, v64 = bnsh_expand_for_gqa(q64, k64, v64)

    batch, heads, q_len, head_dim = q64.shape
    kv_len = k64.shape[2]
    scale = head_dim ** -0.5
    output = torch.empty((batch, heads, q_len, head_dim), dtype=torch.float64, device=q64.device)

    kv_t = k64.transpose(-2, -1).contiguous()
    kv_idx = torch.arange(kv_len, device=q64.device) if is_causal else None
    for b in range(batch):
        for h in range(heads):
            k_head_t = kv_t[b, h]
            v_head = v64[b, h]
            for start in range(0, q_len, chunk_size):
                end = min(start + chunk_size, q_len)
                q_chunk = q64[b, h, start:end]
                scores = torch.matmul(q_chunk, k_head_t) * scale
                if is_causal:
                    assert kv_idx is not None
                    q_idx = torch.arange(start, end, device=q64.device)
                    mask = q_idx[:, None] >= kv_idx[None, :]
                    scores = scores.masked_fill(~mask, -torch.inf)
                probs = torch.softmax(scores, dim=-1)
                output[b, h, start:end] = torch.matmul(probs, v_head)
                del q_chunk, scores, probs

    return output


def compute_rmse(reference: torch.Tensor, test: torch.Tensor) -> float:
    diff = reference - test.to(torch.float64)
    return diff.pow(2).mean().sqrt().item()


def summarize_values(values: list[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "samples": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p01": float(np.percentile(arr, 1)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def choose_plot_xlim(
    methods: list[tuple[str, list[float]]],
    percentile_range: tuple[float, float] | None,
    padding_frac: float,
) -> tuple[float, float] | None:
    if percentile_range is None:
        return None

    pooled = [value for _, values in methods for value in values]
    if not pooled:
        return None

    lower_pct, upper_pct = percentile_range
    if not (0.0 <= lower_pct < upper_pct <= 100.0):
        raise ValueError("plot percentile range must satisfy 0 <= low < high <= 100")

    arr = np.asarray(pooled, dtype=np.float64)
    low, high = np.percentile(arr, [lower_pct, upper_pct])
    if not np.isfinite(low) or not np.isfinite(high):
        return None

    span = high - low
    if span <= 0:
        center = float(low)
        pad = max(abs(center) * max(padding_frac, 1e-3), 1e-8)
        return center - pad, center + pad

    pad = span * max(padding_frac, 0.0)
    return float(low - pad), float(high + pad)


def prepare_qkv_for_operator(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    operator_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if operator_name == "prefill_gqa" or q.shape[0] == k.shape[0]:
        return q.contiguous(), k.contiguous(), v.contiguous()

    q_batched, k_batched, v_batched = bnsh_expand_for_gqa(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
    )
    return (
        q_batched[0].contiguous(),
        k_batched[0].contiguous(),
        v_batched[0].contiguous(),
    )


def save_rmse_plot(
    methods: list[tuple[str, list[float]]],
    output_path: Path,
    _model_id: str,
    _operator_name: str,
    xlim: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, values_list in methods:
        if not values_list:
            continue
        values = np.asarray(values_list, dtype=np.float64)
        color = METHOD_COLORS.get(label)
        if sns is not None and len(values) >= 2 and float(np.ptp(values)) > 0:
            sns.kdeplot(
                values,
                label=label,
                ax=ax,
                fill=True,
                alpha=0.3,
                bw_adjust=0.5,
                color=color,
            )
        else:
            bins = min(10, max(1, len(values)))
            ax.hist(values, bins=bins, density=True, histtype="step", label=label, color=color)

    ax.set_xlabel("Root mean square absolute error")
    ax.set_ylabel("Density")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def save_results_json(
    output_path: Path,
    *,
    model_id: str,
    dataset_name: str,
    dataset_config: str,
    operator_name: str,
    seq_len: int,
    num_runs: int,
    cache_path: Path,
    baselines: list[str],
    xlim: tuple[float, float] | None,
    results: dict[str, list[float]],
) -> None:
    payload = {
        "model_id": model_id,
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "operator_name": operator_name,
        "seq_len": seq_len,
        "num_runs": num_runs,
        "cache_path": str(cache_path),
        "baselines": baselines,
        "plot_xlim": list(xlim) if xlim is not None else None,
        "methods": {
            label: {
                "stats": summarize_values(values),
                "values": [float(value) for value in values],
            }
            for label, values in results.items()
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_error_experiment(
    samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    operator_name: str,
    baseline_tokens: list[str],
    flex_max_autotune: bool,
    reference_chunk_size: int,
) -> dict[str, list[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_causal = operator_is_causal(operator_name)
    specs = build_method_specs(operator_name, baseline_tokens, flex_max_autotune)
    methods: list[ActiveMethod] = []
    for spec in specs:
        try:
            runner = spec.factory()
        except Exception as exc:
            logger.warning("%s factory failed with exception: %s", spec.label, exc)
            continue
        methods.append(ActiveMethod(spec.key, spec.label, runner, []))

    if not methods:
        raise RuntimeError("No methods were initialized successfully.")

    sample_q_heads = samples[0][0].shape[0]
    sample_kv_heads = samples[0][1].shape[0]
    if operator_name != "prefill_gqa" and sample_q_heads != sample_kv_heads:
        logger.info(
            "Expanding KV heads from %d to %d so Qwen-derived GQA samples can run as %s",
            sample_kv_heads,
            sample_q_heads,
            operator_name,
        )

    for sample_idx, (q_cpu, k_cpu, v_cpu) in enumerate(
        tqdm.tqdm(samples, desc="Running error analysis"),
        start=1,
    ):
        q_cpu_prepared, k_cpu_prepared, v_cpu_prepared = prepare_qkv_for_operator(
            q_cpu,
            k_cpu,
            v_cpu,
            operator_name,
        )
        q = q_cpu_prepared.unsqueeze(0).to(device=device)
        k = k_cpu_prepared.unsqueeze(0).to(device=device)
        v = v_cpu_prepared.unsqueeze(0).to(device=device)

        reference = naive_attention_fp64(
            q,
            k,
            v,
            is_causal=is_causal,
            chunk_size=reference_chunk_size,
        )

        for method in methods:
            if not method.enabled:
                continue
            try:
                supported = method.runner.setup(q, k, v)
            except Exception as exc:
                logger.warning("%s setup failed on sample %d: %s", method.label, sample_idx, exc)
                method.enabled = False
                try:
                    method.runner.free_members_memory()
                except Exception:
                    pass
                continue

            if not supported:
                logger.warning("%s does not support operator %s", method.label, operator_name)
                method.enabled = False
                try:
                    method.runner.free_members_memory()
                except Exception:
                    pass
                continue

            try:
                method.runner.run()
                output = method.runner.get_output()
                if output.shape != reference.shape:
                    raise RuntimeError(
                        f"shape mismatch: got {tuple(output.shape)}, expected {tuple(reference.shape)}"
                    )
                method.values.append(compute_rmse(reference, output))
            except Exception as exc:
                logger.warning("%s failed on sample %d: %s", method.label, sample_idx, exc)
                method.enabled = False
            finally:
                try:
                    method.runner.free_members_memory()
                except Exception:
                    pass

        del q, k, v, reference
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results = {method.label: method.values for method in methods if method.values}
    if "Nautilus" not in results:
        raise RuntimeError("Nautilus did not produce any valid samples.")
    return results


def main(argv: list[str] | None = None) -> ExperimentSummary:
    args = parse_args(argv)
    configure_logging(None)
    ensure_triton_cache_dir()

    payload = get_real_qkv_payload(
        num_samples=args.num_runs,
        seq_len=args.seq_len,
        cache_path=args.cache_path.expanduser(),
        model_id=args.model_id,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        seed=args.seed,
    )

    samples = payload["samples"][: args.num_runs]
    if not samples:
        raise RuntimeError("No samples available.")

    requested_operator = normalize_operator_name(args.operator)
    operator_name = infer_operator_name(
        requested_operator,
        samples[0][0],
        samples[0][1],
        is_causal=payload["is_causal"],
    )
    output_path, results_output, log_path = resolve_artifact_paths(
        operator_name,
        output=args.output,
        results_output=args.results_output,
        log_file=args.log_file,
    )
    configure_logging(log_path)
    baseline_tokens = parse_baseline_tokens(args.baselines)
    logger.info("Using operator %s for model %s", operator_name, payload["model_id"])
    logger.info("Artifacts: plot=%s results=%s log=%s", output_path, results_output, log_path)

    results = run_error_experiment(
        samples,
        operator_name=operator_name,
        baseline_tokens=baseline_tokens,
        flex_max_autotune=args.flex_max_autotune,
        reference_chunk_size=args.reference_chunk_size,
    )

    ordered_methods: list[tuple[str, list[float]]] = []
    for key in ["nautilus", *baseline_tokens]:
        label = METHOD_LABELS.get(key)
        if label and label in results:
            ordered_methods.append((label, results[label]))

    percentile_range = None
    if args.plot_percentile_range is not None:
        percentile_range = tuple(args.plot_percentile_range)
    xlim = choose_plot_xlim(
        ordered_methods,
        percentile_range=percentile_range,
        padding_frac=args.plot_padding_frac,
    )
    save_rmse_plot(ordered_methods, output_path, payload["model_id"], operator_name, xlim=xlim)

    save_results_json(
        results_output,
        model_id=payload["model_id"],
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        operator_name=operator_name,
        seq_len=args.seq_len,
        num_runs=args.num_runs,
        cache_path=args.cache_path.expanduser(),
        baselines=baseline_tokens,
        xlim=xlim,
        results=results,
    )

    for label, values in results.items():
        stats = summarize_values(values)
        logger.info(
            "%s: mean %.4e median %.4e min %.4e max %.4e p01 %.4e p99 %.4e samples=%d",
            label,
            stats["mean"],
            stats["median"],
            stats["min"],
            stats["max"],
            stats["p01"],
            stats["p99"],
            stats["samples"],
        )
    if xlim is not None:
        logger.info("Using x-axis limits [%.4e, %.4e]", xlim[0], xlim[1])
    logger.info("Wrote RMSE plot to %s", output_path)
    logger.info("Wrote RMSE results to %s", results_output)

    return ExperimentSummary(
        operator_name=operator_name,
        output_path=output_path,
        methods=results,
    )


if __name__ == "__main__":
    main()
