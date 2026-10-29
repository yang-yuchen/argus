"""Plot normalized performance from Argus_Benchmark.csv (Excel file)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parents[1]

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def to_power_of_2_label(value):
    """Convert a number to power of 2 notation (e.g., 1024 -> '2^{10}')"""
    if value <= 0:
        return str(int(value))
    
    # Calculate log2
    exponent = math.log2(value)
    
    # Check if it's a perfect power of 2
    if abs(exponent - round(exponent)) < 1e-9:  # Close enough to integer
        return f'$2^{{{int(round(exponent))}}}$'
    else:
        # If not a perfect power of 2, just return the original value
        return str(int(value))

# Color mapping for methods (matching the style from plot_gh200.py)
ALL_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
green_idx = 2  # The default green is at index 2
ALL_COLORS = [ALL_COLORS[green_idx]] + ALL_COLORS[:green_idx] + ALL_COLORS[green_idx + 1 :]

# Method display names and colors
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
    'TVM': 'TVM',
    'Ours': 'Nautilus',
    # FP8 variants
    'FlexAttn_fp8': 'FlexAttn FP8',
    'TAWA_fp8': 'TAWA FP8',
    'Ours_fp8': 'Nautilus FP8',
    # BF16 variants
    'FlexAttn_bf16': 'FlexAttn BF16',
    'TAWA_bf16': 'TAWA BF16',
    'Ours_bf16': 'Nautilus BF16',
}

# Tensor compilers
TENSOR_COMPILERS = {
    'Ours',           # Nautilus
    'TAWA',           # Tawa
    'TileLang',       # TileLang
    'Helion',         # Helion
    'OpenAITriton',   # Triton
    'FlexAttn',       # FlexAttention
    'TVM',            # TVM
    'TriDaoTriton',   # Tri-Dao Triton
}

# FP16 baselines (non-tensor compiler methods)
FP16_BASELINES = {
    'Ours',              # Nautilus
    'ThunderKittens',    # ThunderKittens
    'Torch_CUDNN',       # cuDNN
    'Torch_Flash',       # PyTorch Flash
    'FlashInfer-cutlass', # FlashInfer
    'TriDao-cutlass',    # Tri-Dao CUTLASS
}

# Operator display names
OPERATOR_DISPLAY_NAMES = {
    'PF_GLOBAL': 'Global Attention',
    'PF_CAUSAL': 'Causal Attention',
    'PF_GQA': 'GQA Attention',
    'PF_ALIBI': 'ALiBi (Prefill)',
    'PF_SOFTCAP': 'SoftCap (Prefill)',
    'PF_WINDOWED': 'Windowed Attention (Prefill)',
    'DC_CAUSAL': 'Causal Attention (Decode)',
    'DC_GQA': 'GQA (Decode)',
    'DC_ALIBI': 'ALiBi (Decode)',
    'DC_SOFTCAP': 'SoftCap (Decode)',
}

def build_colormap(methods: list[str]) -> dict[str, str]:
    """Build a color map for methods.

    Our methods use warm, bold colors. Baselines use cooler, lighter colors.
    """
    # Define explicit color mapping
    METHOD_COLORS = {
        # Our methods - warm, bold colors
        'Nautilus': '#2ca02c',      # green (swapped with Helion)
        'TAWA': '#ff7f0e',          # orange
        'Helion': '#d62728',        # bold red (swapped with Nautilus)
        'TileLang': '#9467bd',      # purple
        'TileIR': '#8c564b',        # brown
        'JAX Mosaic': '#e377c2',    # pink/magenta
        'TVM': '#e377c2',           # pink/magenta

        # FP8 variants
        'FlexAttn FP8': '#7f7f7f',  # gray
        'TAWA FP8': '#ff7f0e',      # orange
        'Nautilus FP8': '#2ca02c',  # green (swapped with Helion)

        # BF16 variants
        'FlexAttn BF16': '#7f7f7f', # gray
        'TAWA BF16': '#ff7f0e',     # orange
        'Nautilus BF16': '#2ca02c', # green (swapped with Helion)

        # Baselines - cooler, lighter colors
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

    # Fallback colors if we have methods not in the predefined map
    fallback_colors = ['#556b2f', '#4682b4', '#cd5c5c', '#6495ed', '#daa520',
                      '#40e0d0', '#ee82ee', '#f4a460', '#5f9ea0', '#d2691e']
    fallback_idx = 0

    for method in methods:
        if method in METHOD_COLORS:
            colormap[method] = METHOD_COLORS[method]
            used_colors.add(METHOD_COLORS[method])
        else:
            # Assign a fallback color that hasn't been used
            while fallback_idx < len(fallback_colors) and fallback_colors[fallback_idx] in used_colors:
                fallback_idx += 1
            if fallback_idx < len(fallback_colors):
                colormap[method] = fallback_colors[fallback_idx]
                used_colors.add(fallback_colors[fallback_idx])
                fallback_idx += 1
            else:
                # If we run out, just cycle through matplotlib defaults
                colormap[method] = ALL_COLORS[len(colormap) % len(ALL_COLORS)]

    return colormap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot normalized performance from Excel benchmark data")
    parser.add_argument(
        "--input-fp16",
        type=Path,
        default=ROOT / "data/5090_latency_us_new.csv",
        help="Path to the FP16 CSV benchmark file",
    )
    parser.add_argument(
        "--input-fp8",
        type=Path,
        default=ROOT / "data/datasheet_fp8.csv",
        help="Path to the FP8 CSV benchmark file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "figures",
        help="Directory where PDFs will be written",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="Ours",
        help="Method to use as baseline for normalization",
    )
    parser.add_argument(
        "--operators",
        nargs="*",
        default=None,
        help="Specific operators to plot (default: all)",
    )
    return parser.parse_args()


def load_data(file_path: Path, sheet_name: str | None = None) -> pd.DataFrame:
    """Load data from CSV or Excel file."""
    if file_path.suffix.lower() == '.csv':
        # Load CSV file
        df = pd.read_csv(file_path)
        logger.info(f"Loaded CSV file: {file_path}")
        
        # Normalize column names
        if 'operator' in df.columns:
            df = df.rename(columns={'operator': 'Operator', 'seq_len': 'SeqLen'})
            # Map operator names to standard format
            operator_map = {
                'Global (PF)': 'PF_GLOBAL',
                'Causal (PF)': 'PF_CAUSAL',
                'GQA (PF)': 'PF_GQA',
            }
            df['Operator'] = df['Operator'].map(operator_map).fillna(df['Operator'])
            # Map method names to standard format
            method_map = {
                'FlexAttn': 'FlexAttn',
                'cuDNN': 'Torch_CUDNN',
                'OpenAI Triton': 'OpenAITriton',
                'Tri-Dao Triton': 'TriDaoTriton',
                'PyTorch': 'Torch_Flash',
                'FlashInfer': 'FlashInfer',
                'tilelang': 'TileLang',
                'tri-dao': 'TriDao-cutlass',
                'Ours': 'Ours',
            }
            df = df.rename(columns=method_map)
    else:
        # Load Excel file
        if sheet_name:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            # Load first sheet
            excel_data = pd.read_excel(file_path, sheet_name=None)
            first_sheet = list(excel_data.keys())[0]
            df = excel_data[first_sheet]
            logger.info(f"Using sheet: {first_sheet}")

    return df


def normalize_by_baseline(df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Normalize all method columns by the baseline method.

    Returns relative performance where baseline = 1.0.
    Higher values = better performance (faster).
    """
    df = df.copy()

    # Get all method columns (exclude Operator and SeqLen)
    method_cols = [col for col in df.columns if col not in ['Operator', 'SeqLen']]

    if baseline not in method_cols:
        raise ValueError(f"Baseline '{baseline}' not found in columns: {method_cols}")

    # Convert all method columns to numeric type (in case they were loaded as strings)
    for col in method_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # IMPORTANT: Save the baseline values BEFORE normalization to avoid using normalized values
    baseline_values = df[baseline].copy()

    # Normalize: baseline / method (inverted to show relative performance)
    # This makes baseline = 1.0, slower methods < 1.0, faster methods > 1.0
    for col in method_cols:
        df[col] = baseline_values / df[col]

    return df


def plot_multi_operator_row(df: pd.DataFrame, operators: list[str], baseline: str,
                           output_dir: Path, methods_filter: set[str] | None = None,
                           suffix: str = "", show_legend: bool = True) -> None:
    """Create a row of subfigures for multiple operators.

    Args:
        df: DataFrame with normalized data
        operators: List of operator names to plot
        baseline: Baseline method name
        output_dir: Directory to save plots
        methods_filter: Set of method names to include (if None, include all)
        suffix: Suffix to add to filename
        show_legend: Whether to show legend at the top
    """
    num_ops = len(operators)
    # Reduced height from 5 to 3.5 for more compact plots
    fig, axes = plt.subplots(1, num_ops, figsize=(6 * num_ops, 3.5))

    if num_ops == 1:
        axes = [axes]

    # Collect all methods across operators for consistent color mapping
    all_method_cols = set()
    for operator in operators:
        op_df = df[df['Operator'] == operator].copy()
        if not op_df.empty:
            method_cols = [col for col in op_df.columns if col not in ['Operator', 'SeqLen']]
            if methods_filter is not None:
                method_cols = [col for col in method_cols if col in methods_filter]
            all_method_cols.update(method_cols)

    # Build color map for all methods
    display_methods = [METHOD_DISPLAY_NAMES.get(m, m) for m in all_method_cols]
    colormap = build_colormap(display_methods)

    # First pass: find global min value across all operators to determine y-axis range
    global_min = float('inf')
    for operator in operators:
        op_df = df[df['Operator'] == operator].copy()
        if not op_df.empty:
            method_cols = [col for col in op_df.columns if col not in ['Operator', 'SeqLen']]
            if methods_filter is not None:
                method_cols = [col for col in method_cols if col in methods_filter]
            for method_col in method_cols:
                valid_values = op_df[method_col].dropna()
                if not valid_values.empty:
                    global_min = min(global_min, valid_values.min())

    # All plots start from 0
    y_lower_bound = 0

    for idx, (operator, ax) in enumerate(zip(operators, axes)):
        # Filter data for this operator
        op_df = df[df['Operator'] == operator].copy()

        if op_df.empty:
            logger.warning(f"No data found for operator: {operator}")
            continue

        # Sort by sequence length
        op_df = op_df.sort_values('SeqLen')

        # Get method columns (exclude Operator and SeqLen)
        all_cols = [col for col in op_df.columns if col not in ['Operator', 'SeqLen']]

        # Filter methods if specified
        if methods_filter is not None:
            method_cols = [col for col in all_cols if col in methods_filter]
        else:
            method_cols = all_cols

        if not method_cols:
            logger.warning(f"No methods to plot for operator: {operator}")
            continue

        # Sort methods to ensure 'Ours' (Nautilus) appears first in the legend
        def sort_key(method):
            if method == 'Ours':
                return (0, method)  # Ours first
            elif method.endswith('_fp8') and method.replace('_fp8', '') == 'Ours':
                return (0, method)  # Ours_fp8 first
            elif method.endswith('_bf16') and method.replace('_bf16', '') == 'Ours':
                return (0, method)  # Ours_bf16 first
            else:
                return (1, method)  # Others after
        
        method_cols_sorted = sorted(method_cols, key=sort_key)

        # Plot each method
        for method_col in method_cols_sorted:
            # Skip if all values are NaN
            if op_df[method_col].isna().all():
                continue

            display_name = METHOD_DISPLAY_NAMES.get(method_col, method_col)
            color = colormap.get(display_name, None)

            # Plot with markers
            ax.plot(
                op_df['SeqLen'],
                op_df[method_col],
                marker='o',
                label=display_name,
                color=color,
                linewidth=2,
                markersize=6,
            )

        # Styling
        ax.set_xlabel('Sequence Length', fontsize=20)
        if idx == 0:
            ax.set_ylabel('Relative Performance', fontsize=20)

        operator_name = OPERATOR_DISPLAY_NAMES.get(operator, operator)
        ax.set_title(f'{operator_name}', fontsize=22)

        # Set x-axis to log scale for better visualization
        ax.set_xscale('log', base=2)

        # Format x-axis ticks to show power of 2 notation
        seq_lens = op_df['SeqLen'].unique()
        ax.set_xticks(seq_lens)
        ax.set_xticklabels([to_power_of_2_label(sl) for sl in seq_lens], fontsize=16)
        
        # Set y-axis tick label fontsize
        ax.tick_params(axis='y', labelsize=16)

        # Grid
        ax.grid(True, linestyle='--', alpha=0.5)

        # Set y-axis to start from calculated lower bound (0 or 0.35)
        ax.set_ylim(bottom=y_lower_bound)

    # Add legend at the top if requested
    if show_legend and method_cols:
        # Get handles and labels from the first axis
        handles, labels = axes[0].get_legend_handles_labels()
        # Moved legend higher (1.15 instead of 1.08) to avoid overlap with titles
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15),
                  fontsize=18, frameon=True, ncol=len(labels))

    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"rtx5090{suffix}.pdf"
    out_path = output_dir / filename
    plt.tight_layout()
    if show_legend:
        # Reduced top margin (0.80 instead of 0.85) to make more room for legend
        plt.subplots_adjust(top=0.80)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def main() -> None:
    try:
        import coloredlogs  # type: ignore
        coloredlogs.install(level=logging.INFO, fmt="%(asctime)s %(levelname)s %(message)s")
    except Exception:  # pragma: no cover
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args = parse_args()

    # Define the 3 operators for prefill comparison
    prefill_operators = ['PF_GLOBAL', 'PF_CAUSAL', 'PF_GQA']

    # ========== FP16 Data ==========
    # Load FP16 data
    logger.info(f"Loading FP16 data from {args.input_fp16}")
    df_fp16 = load_data(args.input_fp16)

    # Show available operators
    operators = df_fp16['Operator'].unique()
    logger.info(f"Available FP16 operators: {list(operators)}")

    # Normalize FP16 data
    logger.info(f"Normalizing FP16 by baseline: {args.baseline}")
    df_fp16_normalized = normalize_by_baseline(df_fp16, args.baseline)

    # Figure 1: Tensor compiler comparison FP16 (3 subfigures without legend)
    logger.info("Generating Figure 1: Tensor compiler FP16 comparison")
    plot_multi_operator_row(df_fp16_normalized, prefill_operators, args.baseline,
                            args.output_dir, methods_filter=TENSOR_COMPILERS,
                            suffix="-fp16-tc1", show_legend=False)

    logger.info(f"All plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
