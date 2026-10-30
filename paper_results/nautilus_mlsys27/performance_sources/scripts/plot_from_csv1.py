"""
Plot normalized performance from datasheet.csv (Stacked CSV format).
Features:
- Parses raw CSV text to handle multiple vertically stacked tables.
- Assigns GH200 FP16, BF16, FP8 to the partitions in order.
- Normalizes Latency to Relative Performance (Baseline / Method).
- Ensures lines connect the dots.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from io import StringIO

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# --- Configuration Constants ---
COL_OP = 'Operator'
COL_SEQLEN = 'SeqLen'

# Synthetic Columns
COL_DEVICE = 'Device'
COL_DTYPE = 'DataType'

# Order of configs assumed in the stacked CSV tables
CSV_CONFIG_ORDER = [
    ('GH200', 'FP16'),
    ('GH200', 'BF16'),
    ('GH200', 'FP8'),
]

# Color mapping
ALL_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
green_idx = 2
ALL_COLORS = [ALL_COLORS[green_idx]] + ALL_COLORS[:green_idx] + ALL_COLORS[green_idx + 1 :]

# Method display names
METHOD_DISPLAY_NAMES = {
    'FlexAttn': 'FlexAttn',
    'Torch_Flash': 'PyTorch Flash',
    'Torch_CUDNN': 'cuDNN',
    'OpenAITriton': 'OpenAI Triton',
    'TriDaoTriton': 'Tri-Dao Triton',
    'TriDao-cutlass': 'Tri-Dao Cutlass',
    'FlashInfer-cutlass': 'FlashInfer',
    'TileIR': 'TileIR',
    'TileLang': 'TileLang',
    'ThunderKittens': 'ThunderKittens',
    'TAWA': 'TAWA',
    'Helion': 'Helion',
    'JAX Mosaic': 'JAX Mosaic',
    'Neptune+Tawa': 'Ours',
    'Ours': 'Ours', # Map 'Ours' column to 'Ours' label
}

# Operator display names
OPERATOR_DISPLAY_NAMES = {
    'PF_GLOBAL': 'Global Attention (Prefill)',
    'PF_CAUSAL': 'Causal Attention (Prefill)',
    'PF_GQA': 'GQA (Prefill)',
    'PF_ALIBI': 'ALiBi (Prefill)',
    'PF_SOFTCAP': 'SoftCap (Prefill)',
    'PF_WINDOWED': 'Windowed Attention (Prefill)',
    'DC_CAUSAL': 'Causal Attention (Decode)',
    'DC_GQA': 'GQA (Decode)',
    'DC_ALIBI': 'ALiBi (Decode)',
    'DC_SOFTCAP': 'SoftCap (Decode)',
}

def build_colormap(methods: list[str]) -> dict[str, str]:
    """Build a color map for methods."""
    METHOD_COLORS = {
        'Ours': '#d62728',          # bold red
        'Neptune+Tawa': '#d62728',  # bold red (backward compat)
        'TAWA': '#ff7f0e',          # orange
        'Helion': '#2ca02c',        # green
        'TileLang': '#9467bd',      # purple
        'TileIR': '#8c564b',        # brown
        'JAX Mosaic': '#e377c2',    # pink/magenta
        'cuDNN': '#1f77b4',         # blue
        'PyTorch Flash': '#17becf', # cyan/teal
        'FlexAttn': '#7f7f7f',      # gray
        'FlashInfer': '#bcbd22',    # olive/yellow-green
        'Tri-Dao Cutlass': '#aec7e8',   # light blue
        'Tri-Dao Triton': '#98df8a',    # light green
        'OpenAI Triton': '#c5b0d5',     # light purple
        'ThunderKittens': '#c49c94',    # tan/beige
    }

    colormap: dict[str, str] = {}
    used_colors = set()
    # Fallback colors
    fallback_colors = ['#556b2f', '#4682b4', '#cd5c5c', '#6495ed', '#daa520',
                      '#40e0d0', '#ee82ee', '#f4a460', '#5f9ea0', '#d2691e']
    fallback_idx = 0

    for method in methods:
        display_name = METHOD_DISPLAY_NAMES.get(method, method)
        
        if display_name in METHOD_COLORS:
            colormap[method] = METHOD_COLORS[display_name]
            used_colors.add(METHOD_COLORS[display_name])
        else:
            while fallback_idx < len(fallback_colors) and fallback_colors[fallback_idx] in used_colors:
                fallback_idx += 1
            if fallback_idx < len(fallback_colors):
                colormap[method] = fallback_colors[fallback_idx]
                used_colors.add(fallback_colors[fallback_idx])
                fallback_idx += 1
            else:
                colormap[method] = ALL_COLORS[len(colormap) % len(ALL_COLORS)]
    return colormap

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot normalized performance from CSV benchmark data")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasheet.csv"),
        help="Path to the benchmark file (csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots_custom"),
        help="Directory where PDFs will be written",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="Ours", 
        help="Method column to use as baseline for normalization",
    )
    return parser.parse_args()

def load_data(file_path: Path) -> pd.DataFrame:
    """
    Manually parses a raw text CSV that contains multiple stacked tables.
    The first table has a standard header with 'Operator'.
    Subsequent tables are identified by rows containing dtype suffixes (_bf16, _fp8).
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Reading raw text from {file_path}...")

    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    # Find table boundaries by analyzing the CSV structure
    # Table 1: Starts with "Operator" header (FP16)
    # Table 2: Row with _bf16 suffix in method names (BF16)
    # Table 3: Row with _fp8 suffix in method names (FP8)

    table_ranges = []

    # Find FP16 table (standard header)
    fp16_start = None
    for i, line in enumerate(lines):
        clean_line = line.strip()
        if clean_line.startswith('Operator') or clean_line.startswith('"Operator"') or clean_line.startswith('\ufeffOperator'):
            fp16_start = i
            break

    if fp16_start is None:
        raise ValueError("Could not find 'Operator' header for FP16 table")

    # Find BF16 table (look for _bf16 suffix)
    bf16_start = None
    for i in range(fp16_start + 1, len(lines)):
        if '_bf16' in lines[i].lower():
            bf16_start = i
            break

    # Find FP8 table (look for _fp8 suffix)
    fp8_start = None
    if bf16_start:
        for i in range(bf16_start + 1, len(lines)):
            if '_fp8' in lines[i].lower():
                fp8_start = i
                break

    # Determine ranges
    if fp16_start is not None:
        if bf16_start is not None:
            table_ranges.append((fp16_start, bf16_start, 'FP16'))
            if fp8_start is not None:
                table_ranges.append((bf16_start, fp8_start, 'BF16'))
                # FP8 goes to end of data
                end_idx = len(lines)
                # Find where data actually ends (skip empty lines at end)
                for i in range(fp8_start, len(lines)):
                    if not lines[i].strip() or lines[i].count(',') < 2:
                        end_idx = i
                        break
                table_ranges.append((fp8_start, end_idx, 'FP8'))
            else:
                # No FP8 found, BF16 goes to end
                table_ranges.append((bf16_start, len(lines), 'BF16'))
        else:
            # No BF16/FP8 found, just FP16
            table_ranges.append((fp16_start, len(lines), 'FP16'))

    logger.info(f"Found {len(table_ranges)} tables:")
    for start, end, dtype in table_ranges:
        logger.info(f"  {dtype}: lines {start+1} to {end}")

    dfs = []

    for idx, (start_idx, end_idx, dtype) in enumerate(table_ranges):
        chunk_lines = lines[start_idx:end_idx]

        # For BF16 and FP8, we need to extract the header from the FP16 section
        if dtype in ['BF16', 'FP8']:
            # Skip the dtype marker row and get to data
            data_start = 0
            for i, line in enumerate(chunk_lines):
                # Find first row that starts with "PF_" (actual data)
                if line.strip().startswith('PF_'):
                    data_start = i
                    break

            # Use the FP16 header as template
            fp16_header = lines[fp16_start]

            # Extract just the data rows
            data_lines = chunk_lines[data_start:]

            # Combine header with data
            chunk_lines = [fp16_header] + data_lines

        # Remove empty lines
        chunk_lines = [l for l in chunk_lines if l.strip() and l.count(',') > 2]

        if not chunk_lines:
            logger.warning(f"No data found for {dtype}")
            continue

        # Parse this chunk as a CSV
        chunk_str = "".join(chunk_lines)
        try:
            df_chunk = pd.read_csv(StringIO(chunk_str))
        except Exception as e:
            logger.warning(f"Failed to parse {dtype} chunk: {e}")
            continue

        # Clean columns (remove spaces and BOM)
        df_chunk.columns = df_chunk.columns.str.strip().str.replace('\ufeff', '')

        # Filter to only keep rows with valid operators
        df_chunk = df_chunk[df_chunk[COL_OP].str.startswith('PF_', na=False)]

        # Assign Device/Dtype
        device = 'GH200'
        df_chunk[COL_DEVICE] = device
        df_chunk[COL_DTYPE] = dtype
        logger.info(f"Table {idx+1}: Assigned {device} {dtype} ({len(df_chunk)} rows)")

        dfs.append(df_chunk)

    if not dfs:
        raise ValueError("No valid data parsed.")

    full_df = pd.concat(dfs, ignore_index=True)

    # Ensure numeric columns are actually numeric
    for col in full_df.columns:
        if col not in [COL_OP, COL_DEVICE, COL_DTYPE]:
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce')

    return full_df

def normalize_by_baseline(df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """
    Normalize method columns.
    Assumes data is LATENCY (lower is better).
    Relative Performance = Baseline / Method.

    Example:
    Baseline (Ours) = 10ms (fast)
    Method = 20ms (slower than baseline)
    Result = 10 / 20 = 0.5 (method is 2x slower, worse)

    Method = 5ms (faster than baseline)
    Result = 10 / 5 = 2.0 (method is 2x faster, better)

    Baseline always becomes 1.0. Higher values = better performance.
    This matches argus/plot.py:433 convention.
    """
    df = df.copy()

    exclude_cols = [COL_OP, COL_SEQLEN, COL_DEVICE, COL_DTYPE]
    existing_cols = set(df.columns)
    method_cols = [col for col in existing_cols if col not in exclude_cols]

    # Resolve Baseline
    real_baseline = baseline
    if baseline not in df.columns:
        # Try to match display name back to column
        for col, display in METHOD_DISPLAY_NAMES.items():
            if display == baseline and col in df.columns:
                real_baseline = col
                break

    if real_baseline not in df.columns:
        logger.warning(f"Baseline '{baseline}' not found. Skipping normalization.")
        return df

    logger.info(f"Normalizing using baseline: {real_baseline}")

    # DEBUG: Print baseline values before normalization
    print("\n" + "="*80)
    print("DEBUG: BEFORE NORMALIZATION")
    print(f"Baseline column: {real_baseline}")
    print(f"Method columns: {method_cols}")
    sample_rows = df.head(3)
    print(f"\nFirst 3 rows of baseline ({real_baseline}):")
    print(sample_rows[[COL_OP, COL_SEQLEN, real_baseline]])
    if 'FlexAttn' in df.columns:
        print(f"\nFirst 3 rows of FlexAttn:")
        print(sample_rows[[COL_OP, COL_SEQLEN, 'FlexAttn']])
    if 'TAWA' in df.columns:
        print(f"\nFirst 3 rows of TAWA:")
        print(sample_rows[[COL_OP, COL_SEQLEN, 'TAWA']])
    print("="*80 + "\n")

    # Save baseline values BEFORE normalization
    baseline_values = df[real_baseline].copy()

    for col in method_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                # Baseline / Method (baseline becomes 1.0, lower latency = higher rel_perf)
                df[col] = baseline_values / df[col]
            except Exception:
                pass

    # DEBUG: Print after normalization
    print("\n" + "="*80)
    print("DEBUG: AFTER NORMALIZATION")
    sample_rows_after = df.head(3)
    print(f"\nFirst 3 rows of {real_baseline} (should be 1.0):")
    print(sample_rows_after[[COL_OP, COL_SEQLEN, real_baseline]])
    if 'FlexAttn' in df.columns:
        print(f"\nFirst 3 rows of FlexAttn (should be ~0.6-0.8):")
        print(sample_rows_after[[COL_OP, COL_SEQLEN, 'FlexAttn']])
    if 'TAWA' in df.columns:
        print(f"\nFirst 3 rows of TAWA (should be ~0.7-0.9):")
        print(sample_rows_after[[COL_OP, COL_SEQLEN, 'TAWA']])
    print("="*80 + "\n")

    return df

def plot_row_operators(df: pd.DataFrame, 
                      operators: list[str], 
                      device: str, 
                      dtype: str, 
                      output_path: Path, 
                      show_legend: bool = False) -> None:
    
    # 1. Filter Data
    mask = (df[COL_DEVICE] == device) & (df[COL_DTYPE] == dtype)
    sub_df = df[mask].copy()
    
    if sub_df.empty:
        logger.warning(f"No data for {device} {dtype}")
        return

    # 2. Setup Figure
    num_ops = len(operators)
    fig, axes = plt.subplots(1, num_ops, figsize=(6 * num_ops, 5), sharey=False)
    if num_ops == 1:
        axes = [axes]

    global_handles = []
    global_labels = []
    seen_labels = set()

    # 3. Plot
    for i, op in enumerate(operators):
        ax = axes[i]
        
        # Filter Operator
        op_data = sub_df[sub_df[COL_OP] == op].copy()
        
        if op_data.empty:
            ax.set_title(OPERATOR_DISPLAY_NAMES.get(op, op))
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            continue

        op_data = op_data.sort_values(COL_SEQLEN)
        
        # Identify Method Columns (numeric only)
        exclude = [COL_OP, COL_SEQLEN, COL_DEVICE, COL_DTYPE]
        methods = [c for c in op_data.columns if c not in exclude and pd.api.types.is_numeric_dtype(op_data[c])]

        colormap = build_colormap(methods)

        # DEBUG: Print data before plotting for first operator only
        if i == 0 and op == 'PF_GLOBAL':
            print("\n" + "="*80)
            print(f"DEBUG: PLOTTING DATA for {op} ({device} {dtype})")
            print(f"Methods to plot: {methods}")
            for method in ['Ours', 'FlexAttn', 'TAWA']:
                if method in op_data.columns:
                    print(f"\n{method} values (first 5 rows):")
                    print(op_data[[COL_SEQLEN, method]].head(5))
            print("="*80 + "\n")

        for method in methods:
            # Skip empty or all-inf columns
            if op_data[method].isna().all():
                continue
            if np.isinf(op_data[method]).all():
                continue

            display_name = METHOD_DISPLAY_NAMES.get(method, method)
            color = colormap.get(method, None)

            # --- PLOTTING ---
            line, = ax.plot(
                op_data[COL_SEQLEN],
                op_data[method],
                marker='o',
                linestyle='-',  # <--- CRITICAL: Connect points with lines
                linewidth=2,
                markersize=6,
                label=display_name,
                color=color,
                alpha=0.9
            )

            if display_name not in seen_labels:
                seen_labels.add(display_name)
                global_handles.append(line)
                global_labels.append(display_name)

        # Axis Styling
        op_title = OPERATOR_DISPLAY_NAMES.get(op, op)
        ax.set_title(op_title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Sequence Length', fontsize=12)
        
        if i == 0:
            ax.set_ylabel('Relative Performance', fontsize=12)
        
        ax.set_xscale('log', base=2)
        
        # Ticks
        unique_lens = sorted(op_data[COL_SEQLEN].dropna().unique())
        if len(unique_lens) > 0:
            ax.set_xticks(unique_lens)
            
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylim(bottom=0)

    # 4. Global Legend
    if show_legend and global_handles:
        # Sort so Ours is first
        def sort_key(tup):
            lbl = tup[1]
            if lbl == 'Ours': return 0
            if 'TAWA' in lbl: return 1
            return 2

        zipped = sorted(zip(global_handles, global_labels), key=sort_key)
        sorted_handles, sorted_labels = zip(*zipped)

        fig.legend(
            sorted_handles,
            sorted_labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.08),  # Reduced from 1.1 to avoid overlap
            ncol=min(len(sorted_handles), 7),  # Increased from 6 to 7 for more compact layout
            fontsize=10,  # Reduced from 12 to 10
            frameon=False,
            columnspacing=1.0,  # Add spacing control
            handletextpad=0.5   # Reduce space between marker and text
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def main() -> None:
    try:
        import coloredlogs
        coloredlogs.install(level=logging.INFO, fmt="%(asctime)s %(levelname)s %(message)s")
    except Exception:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args = parse_args()

    # 1. Load Data
    try:
        df = load_data(args.input)
    except Exception as e:
        logger.error(f"Critical error loading data: {e}")
        return
    
    # 2. Normalize
    df_norm = normalize_by_baseline(df, args.baseline)

    # 3. Plot Tasks
    target_operators = ['PF_GLOBAL', 'PF_CAUSAL', 'PF_GQA']
    
    tasks = [
        # Task 1: GH200 FP16 (Table 1)
        {
            "device": "GH200", 
            "dtype": "FP16", 
            "legend": True, 
            "filename": "1_gh200_fp16_prefill.pdf"
        },
        # Task 2: GH200 BF16 (Table 2)
        {
            "device": "GH200", 
            "dtype": "BF16", 
            "legend": False, 
            "filename": "2_gh200_bf16_prefill.pdf"
        },
        # Task 3: GH200 FP8 (Table 3)
        {
            "device": "GH200", 
            "dtype": "FP8", 
            "legend": False, 
            "filename": "3_gh200_fp8_prefill.pdf"
        },
    ]

    logger.info("Starting Plot Generation...")
    for task in tasks:
        plot_row_operators(
            df_norm,
            operators=target_operators,
            device=task["device"],
            dtype=task["dtype"],
            output_path=args.output_dir / task["filename"],
            show_legend=task["legend"]
        )
    logger.info("Done.")

if __name__ == "__main__":
    main()