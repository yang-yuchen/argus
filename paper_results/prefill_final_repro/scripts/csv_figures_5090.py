import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "figures"

# -----------------------------------------------------------------------------
# Configuration & Style
# -----------------------------------------------------------------------------
COLORS = {
    'flash_attention_2': '#17becf',    # cyan/teal
    'flex_attention': '#7f7f7f',       # gray
    'nautilus': '#2ca02c',             # green
    'sdpa': '#1f77b4',                 # blue
    'tawa': '#ff7f0e',                 # orange
    'tilelang': '#9467bd',             # purple
}

LEGEND_LABELS = {
    'flash_attention_2': 'FlashAttn-2',
    'flex_attention': 'FlexAttn',
    'nautilus': 'Nautilus',
    'sdpa': 'SDPA',
    'tawa': 'Tawa',
    'tilelang': 'TileLang',
}

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def to_power_of_2_label(value):
    """Convert a number to power of 2 notation (e.g., 1024 -> '2^{10}')"""
    if value <= 0:
        return str(value)
    
    # Calculate log2
    exponent = math.log2(value)
    
    # Check if it's a perfect power of 2
    if exponent == int(exponent):
        return f'$2^{{{int(exponent)}}}$'
    else:
        # If not a perfect power of 2, just return the original value
        return str(value)

# -----------------------------------------------------------------------------
# Data Loading and Processing
# -----------------------------------------------------------------------------
def load_data(csv_path):
    """Load CSV data and separate fp8 and fp16 methods"""
    df = pd.read_csv(csv_path)

    # Parse model name, batch size, and sequence length from the first column
    def parse_config(config_str):
        parts = config_str.split(',')
        model_batch = parts[0]  # e.g., "glm-prefill-1"
        seq_len = parts[1]       # e.g., "1024"

        model_parts = model_batch.split('-')
        model = model_parts[0]   # e.g., "glm"
        batch = model_parts[-1]  # e.g., "1"

        return model, int(batch), int(seq_len)

    df[['model', 'batch', 'seq_len']] = df.apply(
        lambda row: pd.Series(parse_config(row['Model'])), axis=1
    )

    return df

def calculate_geomean_speedup(df, baseline_method='nautilus'):
    """Calculate geometric mean speedup of baseline over other methods"""
    from scipy import stats

    results = {}

    # Get all method columns (excluding metadata columns)
    all_methods = [col for col in df.columns
                   if col not in ['Model', 'model', 'batch', 'seq_len']]

    # Separate baseline and comparison methods
    baseline_col = None
    # First try exact match
    if baseline_method in all_methods:
        baseline_col = baseline_method
    else:
        # Try partial match
        for method in all_methods:
            if baseline_method in method:
                baseline_col = method
                break

    if baseline_col is None:
        print(f"Warning: {baseline_method} not found in data")
        return results

    comparison_methods = [m for m in all_methods if m != baseline_col]

    # Calculate speedup for each comparison method
    for method in comparison_methods:
        # Get valid data points (where both baseline and comparison have values > 0)
        valid_mask = (df[baseline_col] > 0) & (df[method] > 0)

        if valid_mask.sum() == 0:
            continue

        baseline_vals = df.loc[valid_mask, baseline_col]
        comparison_vals = df.loc[valid_mask, method]

        # Calculate speedup ratios
        speedups = baseline_vals / comparison_vals

        # Calculate geometric mean
        geomean = stats.gmean(speedups)

        # Get display name
        display_name = LEGEND_LABELS.get(method, method)

        results[display_name] = geomean

    return results

def print_summary_table(df, baseline='nautilus'):
    """Print summary table with geometric mean speedups"""
    print("\n" + "="*70)
    print("RTX 5090 SUMMARY: Nautilus Geometric Mean Speedup Over Baselines")
    print("="*70)

    # Calculate speedups
    print("\nFP16 Methods:")
    print("-" * 70)
    speedups = calculate_geomean_speedup(df, baseline)
    for method, speedup in sorted(speedups.items(), key=lambda x: -x[1]):
        print(f"  Nautilus vs {method:30s}: {speedup:6.3f}x")

    print("="*70)
    print(f"Aggregated across {len(df)} configurations")
    print(f"(Models: {', '.join(sorted(df['model'].unique()))})")
    print(f"(Batch sizes: {', '.join(map(str, sorted(df['batch'].unique())))})")
    print(f"(Seq lengths: {', '.join(map(str, sorted(df['seq_len'].unique())))})")
    print("="*70 + "\n")

def generate_model_wise_summary(df, output_path='throughput_summary_5090.csv', baseline='nautilus'):
    """Generate model-wise summary table with geometric mean speedups as CSV"""
    models = sorted(df['model'].unique())

    summary_rows = []

    for model in models:
        model_df = df[df['model'] == model]
        row = {'Model': model.upper()}

        # Calculate speedups for this model
        speedups = calculate_geomean_speedup(model_df, baseline)
        for method, speedup in speedups.items():
            row[f"{method}"] = speedup

        summary_rows.append(row)

    # Create DataFrame
    summary_df = pd.DataFrame(summary_rows)

    # Calculate average (geometric mean) across all models for each method
    avg_row = {'Model': 'AVERAGE'}
    for col in summary_df.columns:
        if col != 'Model':
            values = summary_df[col].values
            # Use geometric mean for averaging speedups
            from scipy import stats
            avg_row[col] = stats.gmean(values)

    # Append average row
    summary_df = pd.concat([summary_df, pd.DataFrame([avg_row])], ignore_index=True)

    # Save to CSV
    summary_df.to_csv(output_path, index=False)

    # Print table
    print("\n" + "="*100)
    print("RTX 5090 MODEL-WISE SUMMARY: Nautilus Geometric Mean Speedup Over Baselines")
    print("="*100)
    print(summary_df.to_string(index=False))
    print(f"\nSummary table saved to: {output_path}")
    print("="*100 + "\n")

def plot_subplot(ax, df, methods, title, show_y_label=True, show_legend=False):
    """Plot a single subplot"""
    # Sort methods: nautilus first, then tilelang, tawa, then others
    def sort_methods(method_list):
        sorted_list = []
        # First add nautilus if exists
        if 'nautilus' in method_list:
            sorted_list.append('nautilus')
        # Then add tilelang
        if 'tilelang' in method_list:
            sorted_list.append('tilelang')
        # Then add tawa
        if 'tawa' in method_list:
            sorted_list.append('tawa')
        # Then add the rest in alphabetical order
        remaining = [m for m in method_list if m not in sorted_list]
        sorted_list.extend(sorted(remaining))
        return sorted_list

    methods = sort_methods(methods)

    # Prepare data
    seq_lens = sorted(df['seq_len'].unique())
    x = np.arange(len(seq_lens))

    # Bar dimensions
    total_width = 0.8
    width = total_width / len(methods)

    # Plot bars for each method
    for i, method in enumerate(methods):
        offset = (i - len(methods)/2) * width + width/2

        # Get color and label
        color = COLORS.get(method, '#888888')
        label = LEGEND_LABELS.get(method, method)

        # Get throughput values for this method
        values = [df[df['seq_len'] == sl][method].values[0] if len(df[df['seq_len'] == sl]) > 0 else 0
                  for sl in seq_lens]

        ax.bar(x + offset, values, width, label=label, color=color, zorder=3)

    # Formatting
    ax.set_title(title, fontsize=20, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([to_power_of_2_label(sl) for sl in seq_lens], fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    if show_y_label:
        ax.set_ylabel('Throughput (TFLOPs/s)', fontsize=18)

# -----------------------------------------------------------------------------
# Main Plotting
# -----------------------------------------------------------------------------
def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Load RTX 5090 data
    csv_path_5090 = ROOT / "data/throughput_5090new.csv"
    df_5090 = load_data(csv_path_5090)

    # Filter RTX 5090 data to only batch 1
    df_5090_b1 = df_5090[df_5090['batch'] == 1].copy()

    # Print summary tables for RTX 5090 (batch 1 only)
    print_summary_table(df_5090_b1, baseline='nautilus')
    generate_model_wise_summary(df_5090_b1, output_path=OUTPUT / "throughput_summary_5090.csv", baseline='nautilus')

    # Get all 5 models from 5090 data
    models_5090 = sorted(df_5090_b1['model'].unique())

    # Get methods (only FP16, no FP8)
    methods_5090 = [col for col in df_5090_b1.columns
                     if col not in ['Model', 'model', 'batch', 'seq_len']]

    # Create figure: 1 row x 5 columns (one per model)
    fig, axes = plt.subplots(1, 5, figsize=(25, 3.5))

    # Plot each model
    for i, model in enumerate(models_5090):
        model_df = df_5090_b1[df_5090_b1['model'] == model]

        plot_subplot(
            axes[i],
            model_df,
            methods_5090,
            f'{model.upper()}',
            show_y_label=(i == 0),
            show_legend=False
        )

    # Add common x-axis label
    fig.text(0.5, 0.04, 'Sequence Length', ha='center', fontsize=18)

    plt.tight_layout()
    plt.subplots_adjust(top=0.86, bottom=0.16, wspace=0.20)

    # Add legend at the top center of the entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.98),
        bbox_transform=fig.transFigure,
        fontsize=16,
        frameon=True,
        ncol=min(len(labels), 6)
    )

    # Save figure
    plt.savefig(OUTPUT / "throughput_plot_5090.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Plot saved as {OUTPUT / 'throughput_plot_5090.pdf'}")

if __name__ == '__main__':
    main() 
