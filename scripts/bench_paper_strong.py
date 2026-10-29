"""Hopper baseline search using the existing FP64/RMSE metric and cold-L2 nsys timing."""

import argparse
import json
from pathlib import Path
import platform
import statistics
import sys
import traceback

import torch

from argus.operators import BenchmarkOperator
from argus.paper_baselines.strong_validation import (
    accuracy, input_hash, reference_fp64, validation_inputs,
)
from argus.paper_baselines.workloads import build_exported
from argus.runner import BenchmarkRunner
from argus.utils import is_unrecoverable_cuda_error, profiler_ctx, relaunch_with_profiler


VARIANTS = ("decode", "mtp", "sparse", "sparse_split")


class CallableRunner(BenchmarkRunner):
    def __init__(self, variant, name, function):
        super().__init__(BenchmarkOperator(variant), "strong", impl_name=name)
        self.function = function

    def setup(self, *inputs):
        return True

    def run(self):
        with self.mark_ctx():
            self.output = self.function()

    def get_output(self):
        return self.output


def registry(variant, inputs, family):
    if family == "pdf_dense":
        from argus.operators import QKVInputOp
        from argus.flashinfer import FlashInferRunner
        from argus.helion import HelionRunner
        from argus.pytorch.flex import FlexAttentionRunner
        from argus.pytorch.sdpa import AttentionAlgo, TorchSDPARunner
        from argus.triton_runners import OpenAITritonRunner, TriDaoTritonRunner

        is_decode = variant == "decode"
        op = QKVInputOp(f"paper_{variant}", inputs[0].shape[1], inputs[0].shape[-1],
                        is_causal=False, is_decode=is_decode)
        specs = [
            ("flex", lambda: FlexAttentionRunner(op, max_autotune=True)),
            ("torch_flash", lambda: TorchSDPARunner(op, AttentionAlgo.FLASH, False)),
            ("torch_cudnn", lambda: TorchSDPARunner(op, AttentionAlgo.CUDNN, False)),
            ("openai_triton", lambda: OpenAITritonRunner(op, is_causal=False)),
            ("tridao_triton", lambda: TriDaoTritonRunner(op, is_causal=False)),
            ("helion", lambda: HelionRunner(op)),
            ("flashinfer", lambda: FlashInferRunner(op, is_causal=False,
                                                     is_decode=is_decode)),
        ]
        if is_decode:
            from argus.tawa.decoder import TawaDecodeRunner
            from argus.tilelang_pypi.decode import TileLangDecodeRunner

            specs += [
                ("tawa_decode", lambda: TawaDecodeRunner(op)),
                ("tilelang_decode", lambda: TileLangDecodeRunner(op, is_causal=False)),
            ]
        result = []
        for name, make_runner in specs:
            def factory(make_runner=make_runner, name=name):
                from contextlib import nullcontext

                runner = make_runner()
                if not runner.setup(*inputs):
                    raise RuntimeError(f"{name} does not support this shape")
                # Keep the outer candidate NVTX label; inner Argus runners otherwise hide it.
                runner.mark_ctx = lambda: nullcontext()

                def run():
                    runner.run()
                    return runner.get_output()

                run.source = {"family": "PDF dense baseline", "runner": name,
                              "scope": "current Argus latest environment"}
                return run
            result.append((name, factory))
        return result
    if family == "tilelang":
        from argus.paper_baselines.dense import build_dense_baseline, BASELINE_SOURCE
        from argus.paper_baselines.tilelang_tuned import (
            make_candidates, make_hopper_candidates, make_sm90_split_candidates,
        )

        def default():
            return build_dense_baseline(variant, inputs)

        default.source = BASELINE_SOURCE
        split_candidates = (make_sm90_split_candidates(variant, inputs) if variant == "mtp"
                            else make_candidates(variant, inputs))
        return [("tilelang_default", default), *split_candidates,
                *make_hopper_candidates(variant, inputs)]
    if family == "triton":
        from argus.paper_baselines.triton_dense import make_candidates, make_extra_candidates

        return [*make_candidates(variant, inputs), *make_extra_candidates(variant, inputs)]
    elif family == "libraries":
        from argus.paper_baselines.dense_libraries import make_candidates
    elif family == "sparse":
        from argus.paper_baselines.sparse_strong import make_candidates
    else:
        raise ValueError(f"Unknown candidate family: {family}")
    return make_candidates(variant, inputs)


def environment():
    properties = torch.cuda.get_device_properties(0)
    return {
        "host": platform.node(), "gpu": properties.name, "gpu_uuid": str(properties.uuid),
        "compute_capability": [properties.major, properties.minor],
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "packages": {name: {"version": getattr(sys.modules[name], "__version__", None),
                            "path": getattr(sys.modules[name], "__file__", None)}
                     for name in ("tilelang", "tvm", "triton", "flashinfer", "flash_attn")
                     if name in sys.modules},
    }


def save(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n")


def prepare_nautilus(variant, directory):
    records, runners = [], []
    for case_name, cpu_inputs in validation_inputs(variant):
        print(f"Nautilus accuracy {variant}/{case_name}", flush=True)
        inputs = [x.cuda() for x in cpu_inputs]
        function = build_exported(variant, inputs)
        output = function()
        torch.cuda.synchronize()
        ref = reference_fp64(variant, inputs)
        metrics = accuracy(ref, output, variant)
        torch.save({
            "inputs": cpu_inputs, "nautilus_output": output.detach().cpu(),
            "inputs_sha256": input_hash(cpu_inputs), "nautilus_accuracy": metrics,
        }, directory / f"{variant}_{case_name}.pt")
        records.append({"case": case_name, "accuracy": metrics})
        if case_name == "benchmark":
            if not metrics["passed"]:
                raise AssertionError(f"Nautilus fails the benchmark accuracy gate: {metrics}")
            runners.append(CallableRunner(variant, "nautilus", function))
    return [{"name": "nautilus", "status": "ready", "validation": records,
             "source": "original paper_kernels kernel.so; compatible custom TVM runtime"}], runners


def prepare_candidates(variant, family, directory, metadata_path, metadata):
    cases = []
    oracle = json.loads((directory / f"{variant}_nautilus.json").read_text())
    for case_record in oracle["candidates"][0]["validation"]:
        case_name = case_record["case"]
        payload = torch.load(directory / f"{variant}_{case_name}.pt", weights_only=True,
                             map_location="cpu")
        assert input_hash(payload["inputs"]) == payload["inputs_sha256"]
        inputs = [x.cuda() for x in payload["inputs"]]
        ref = reference_fp64(variant, inputs)
        nautilus = payload["nautilus_output"].cuda()
        cases.append({
            "name": case_name, "inputs": inputs, "reference": ref, "nautilus": nautilus,
            "nautilus_accuracy": accuracy(ref, nautilus, variant),
            "factories": dict(registry(variant, inputs, family)),
        })
    records, runners = [], []
    for name, factory in cases[0]["factories"].items():
        print(f"Checking {variant}/{family}/{name}", flush=True)
        record = {"name": name, "source": getattr(factory, "source", {}), "validation": []}
        try:
            function = factory()
            record["source"] = getattr(function, "source", record["source"])
            for i, case in enumerate(cases):
                call = function if i == 0 else case["factories"][name]()
                output = call()
                torch.cuda.synchronize()
                metrics = accuracy(case["reference"], output, variant)
                pair = accuracy(case["nautilus"].double(), output, variant)
                record["validation"].append({
                    "case": case["name"], "accuracy": metrics, "vs_nautilus": pair,
                    "nautilus_accuracy": case["nautilus_accuracy"],
                })
                if i:
                    del call, output
            reference_pass = all(x["accuracy"]["passed"] for x in record["validation"])
            pair_pass = all(x["vs_nautilus"]["passed"] for x in record["validation"]
                            if x["nautilus_accuracy"]["passed"])
            record["nautilus_failed_cases"] = [x["case"] for x in record["validation"]
                                               if not x["nautilus_accuracy"]["passed"]]
            record["all_cases_equivalent"] = reference_pass and pair_pass and not record[
                "nautilus_failed_cases"]
            if reference_pass and pair_pass:
                record["status"] = "ready"
                runners.append(CallableRunner(variant, name, function))
            else:
                record["status"] = "accuracy_failed"
        except Exception as exc:
            record.update(status="compile_or_runtime_failed", error=traceback.format_exc())
            if is_unrecoverable_cuda_error(exc):
                metadata["candidates"] = records + [record]
                save(metadata_path, metadata)
                raise
        records.append(record)
        metadata["candidates"] = records
        save(metadata_path, metadata)
        print(f"{name}: {record['status']}", flush=True)
    return records, runners


def measure(args, stem):
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("These configurations target Hopper SM90 (H100/GH200)")
    payload = {
        "variant": args.variant, "family": args.family, "stage": args.stage,
        "repeat": args.repeat, "environment": environment(), "candidates": [],
        "accuracy_metric": "scripts.error.compute_rmse against FP64 reference",
        "metric": "sum of GPU kernel durations per cold-L2 NVTX repeat (us)",
        "l2_flush_bytes": 200 * 1024**2, "measurement_complete": False,
    }
    if args.stage == "nautilus":
        records, runners = prepare_nautilus(args.variant, args.directory)
    else:
        records, runners = prepare_candidates(
            args.variant, args.family, args.directory, stem.with_suffix(".json"), payload,
        )
    payload.update(candidates=records, environment=environment())
    save(stem.with_suffix(".json"), payload)
    for runner in runners:
        for _ in range(10):
            runner.run()
    torch.cuda.synchronize()
    if runners:
        with profiler_ctx():
            for runner in runners:
                runner.repeat(args.repeat)


def reduce_profile(stem):
    from argus.stats.nsys import nsys_to_dataframe

    payload = json.loads(stem.with_suffix(".json").read_text())
    ready = [x for x in payload["candidates"] if x["status"] == "ready"]
    if ready:
        raw = nsys_to_dataframe(stem.with_suffix(".nsys-rep"))
        raw = raw[raw["repeatIdx"] != "warmup"]
        raw.to_csv(stem.with_suffix(".csv"), index=False)
        for record in ready:
            data = raw[raw["implementation"] == f"strong({record['name']})"]
            if set(data["repeatIdx"]) != {str(i) for i in range(payload["repeat"])}:
                raise ValueError(f"Missing repeats: {stem}/{record['name']}")
            times = data.groupby("repeatIdx")["latency_us"].sum()
            samples = times.reindex([str(i) for i in range(payload["repeat"])]).tolist()
            record.update(status="measured", samples_us=samples, mean_us=statistics.mean(samples),
                          median_us=statistics.median(samples), min_us=min(samples))
    payload["measurement_complete"] = True
    save(stem.with_suffix(".json"), payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("nautilus", "candidates"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--family", choices=("pdf_dense", "tilelang", "triton", "libraries", "sparse"))
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--profile-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stage == "candidates" and args.family is None:
        parser.error("Candidate stage requires --family")
    args.directory = args.directory.resolve()
    args.directory.mkdir(parents=True, exist_ok=True)
    suffix = "nautilus" if args.stage == "nautilus" else args.family
    stem = args.directory / f"{args.variant}_{suffix}"
    if args.profile_child:
        measure(args, stem)
        return
    if stem.with_suffix(".json").exists() or stem.with_suffix(".nsys-rep").exists():
        parser.error("Output already exists")
    relaunch_with_profiler(
        stem, "nsys", False,
        ["-m", "scripts.bench_paper_strong", *sys.argv[1:], "--profile-child"],
        wait_for_cuda_profiler=True,
    )
    reduce_profile(stem)


if __name__ == "__main__":
    main()
