# Decode to causal MTP to causal prefill overview

[PDF](../../plots_from_csv/gh200-fp16-tc1-attention-transition-20261029.pdf)
[PNG](../../plots_from_csv/gh200-fp16-tc1-attention-transition-20261029.png)

The figure reuses the original plot_from_csv.py method colors and latency
normalization. Green Nautilus=1, gray FlexAttn, purple TileLang; higher is faster.
The categorical x-axis increases Q from decode1 through MTP2,4,8,16,32,64,
then shows full prefill Q=KV. It is not a numeric continuous Q axis.

Each stage uses exactly the same eight KV lengths256–32768, with each length
weighted equally. Large points are geometric means of individually normalized
shape results; faint points show all individual KV values. They are shape
variation, not error bars or confidence intervals. KV65536 is excluded because
existing decode/prefill do not cover it. There are64 total shapes.

## Sources and interpretation

- Decode: existing b7749f3 plot inputs, with the previously validated baselines.
- Causal MTP: full sweep with the four validated optimized replacements.
- Causal prefill: historical datasheet_new_asplos.csv, as requested. The TileLang
  column is the first occurrence, matching the original compiler-filter plot;
  the additional duplicate column is not selected.

All are B1/D128/FP16. Decode and prefill have32 heads; MTP has4. Baseline
implementations/environments also come from the corresponding prior experiments.
This figure supports a comparison across the existing workloads, not a claim
that heads, compiler versions and all other parameters were held constant.
Dashed lines visually connect decode to MTP and MTP to prefill for all three
methods, including Nautilus. MTP itself uses solid lines. No new measurements
or interpolated samples are introduced. The MTP H=4 annotation is omitted from
the figure as requested; the actual shape remains documented here and in the CSV.

The historical prefill table is intentionally distinct from the newer validated
prefill comparison under profiles_gh200_compare_20261026_validated. In that
newer table, the geometric-mean plotted TileLang/Nautilus relative performance
is1.0381 (Nautilus is slower in aggregate), whereas the requested historical table
gives0.8907. These sources must not be described as interchangeable results or
as one unified recent sweep. For a publication claim of a smooth transition,
measure all stages at fixed heads and add intermediate Q values first.

## Reproduce

The exact per-shape input is plot_input.csv; source paths and caveats are in
source_manifest.json. No new GPU jobs were submitted.

```bash
/work/nvme/beqg/yucheny8/envs/argus-gh200-latest/bin/python scripts/plot_attention_transition.py --input profiles_paper_kernels/attention_transition_20261029/plot_input.csv --output plots_from_csv/gh200-fp16-tc1-attention-transition-20261029
```

Validation: pytest tests/test_attention_transition_plot.py -q:4 passed.
