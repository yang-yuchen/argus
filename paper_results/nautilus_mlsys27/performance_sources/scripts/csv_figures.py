import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# -----------------------------------------------------------------------------
# Configuration & Style
# -----------------------------------------------------------------------------
COLORS = {
    'flash_attention_2': '#17becf',    # cyan/teal
    'flex_attention': '#7f7f7f',       # gray
    'neptune_tawa': '#2ca02c',         # green
    'sdpa': '#1f77b4',                 # blue
    'tawa': '#ff7f0e',                 # orange
    'tilelang': '#9467bd',             # purple
}

LEGEND_LABELS = {
    'flash_attention_2': 'FlashAttn-2',
    'flex_attention': 'FlexAttn',
    'neptune_tawa': 'Nautilus',
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

def separate_fp8_fp16(df):
    """Separate FP8 and FP16 methods from the dataframe"""
    fp16_methods = []
    fp8_methods = []

    for col in df.columns:
        if col in ['Model', 'model', 'batch', 'seq_len']:
            continue
        if 'fp8' in col.lower():
            fp8_methods.append(col)
        else:
            fp16_methods.append(col)

    return fp16_methods, fp8_methods

def calculate_geomean_speedup(df, baseline_method='neptune_tawa'):
    """Calculate geometric mean speedup of baseline over other methods"""
    from scipy import stats
    
    results = {}
    
    # Get all method columns (excluding metadata columns)
    all_methods = [col for col in df.columns 
                   if col not in ['Model', 'model', 'batch', 'seq_len']]
    
    # Separate baseline and comparison methods
    baseline_col = None
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
        base_method = method.replace('_fp8', '')
        display_name = LEGEND_LABELS.get(base_method, base_method)
        if '_fp8' in method:
            display_name += ' (FP8)'
        
        results[display_name] = geomean
    
    return results

def print_summary_table(df):
    """Print summary table with geometric mean speedups"""
    print("\n" + "="*70)
    print("SUMMARY: Nautilus Geometric Mean Speedup Over Baselines")
    print("="*70)
    
    # Separate FP16 and FP8 data
    fp16_df = df[[col for col in df.columns 
                  if col in ['Model', 'model', 'batch', 'seq_len'] or 'fp8' not in col.lower()]]
    fp8_df = df[[col for col in df.columns 
                 if col in ['Model', 'model', 'batch', 'seq_len'] or 'fp8' in col.lower()]]
    
    # Calculate speedups for FP16
    print("\nFP16 Methods:")
    print("-" * 70)
    fp16_speedups = calculate_geomean_speedup(fp16_df, 'neptune_tawa')
    for method, speedup in sorted(fp16_speedups.items(), key=lambda x: -x[1]):
        print(f"  Nautilus vs {method:30s}: {speedup:6.3f}x")
    
    # Calculate speedups for FP8
    if 'neptune_tawa_fp8' in df.columns:
        print("\nFP8 Methods:")
        print("-" * 70)
        fp8_speedups = calculate_geomean_speedup(fp8_df, 'neptune_tawa_fp8')
        for method, speedup in sorted(fp8_speedups.items(), key=lambda x: -x[1]):
            print(f"  Nautilus (FP8) vs {method:30s}: {speedup:6.3f}x")
    
    print("="*70)
    print(f"Aggregated across {len(df)} configurations")
    print(f"(Models: {', '.join(sorted(df['model'].unique()))})")
    print(f"(Batch sizes: {', '.join(map(str, sorted(df['batch'].unique())))})")
    print(f"(Seq lengths: {', '.join(map(str, sorted(df['seq_len'].unique())))})")
    print("="*70 + "\n")

def generate_model_wise_summary(df, output_path='throughput_summary.csv'):
    """Generate model-wise summary table with geometric mean speedups as CSV"""
    models = sorted(df['model'].unique())
    
    # Separate FP16 and FP8 columns
    fp16_cols = [col for col in df.columns 
                 if col not in ['Model', 'model', 'batch', 'seq_len'] and 'fp8' not in col.lower()]
    fp8_cols = [col for col in df.columns 
                if col not in ['Model', 'model', 'batch', 'seq_len'] and 'fp8' in col.lower()]
    
    summary_rows = []
    
    for model in models:
        model_df = df[df['model'] == model]
        row = {'Model': model.upper()}
        
        # FP16 speedups
        if 'neptune_tawa' in fp16_cols:
            fp16_model_df = model_df[['model', 'batch', 'seq_len'] + fp16_cols]
            fp16_speedups = calculate_geomean_speedup(fp16_model_df, 'neptune_tawa')
            for method, speedup in fp16_speedups.items():
                row[f"{method}"] = speedup
        
        # FP8 speedups
        if 'neptune_tawa_fp8' in fp8_cols:
            fp8_model_df = model_df[['model', 'batch', 'seq_len'] + fp8_cols]
            fp8_speedups = calculate_geomean_speedup(fp8_model_df, 'neptune_tawa_fp8')
            for method, speedup in fp8_speedups.items():
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
    print("MODEL-WISE SUMMARY: Nautilus Geometric Mean Speedup Over Baselines")
    print("="*100)
    print(summary_df.to_string(index=False))
    print(f"\nSummary table saved to: {output_path}")
    print("="*100 + "\n")

def plot_subplot(ax, df, methods, title, show_y_label=True, show_legend=False):
    """Plot a single subplot"""
    # Sort methods: neptune_tawa first, tawa second, then others
    def sort_methods(method_list):
        sorted_list = []
        # First add neptune_tawa if exists
        for m in method_list:
            if 'neptune_tawa' in m:
                sorted_list.append(m)
        # Then add tawa (but not neptune_tawa)
        for m in method_list:
            if m == 'tawa' or m == 'tawa_fp8':
                sorted_list.append(m)
        # Then add the rest
        for m in method_list:
            if m not in sorted_list:
                sorted_list.append(m)
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

        # Get base method name for color (remove _fp8 suffix if exists)
        base_method = method.replace('_fp8', '')
        color = COLORS.get(base_method, '#888888')

        # Get label for legend
        if '_fp8' in method:
            label = LEGEND_LABELS.get(base_method, base_method) + ' (FP8)'
        else:
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
    # Load data
    csv_path = '/u/yucheny8/neptune-internal/e2e_throughput.csv'
    df = load_data(csv_path)
    
    # Print summary tables
    print_summary_table(df)
    generate_model_wise_summary(df, output_path='throughput_summary.csv')

    # Get models
    models = sorted(df['model'].unique())

    # Separate FP8 and FP16 methods
    fp16_methods, fp8_methods = separate_fp8_fp16(df)

    # Create figure: 5 rows (models) x 4 columns (fp16_batch1, fp16_batch8, fp8_batch1, fp8_batch8)
    fig, axes = plt.subplots(len(models), 4, figsize=(20, 3.3 * len(models)))

    # Plot each model
    for row_idx, model in enumerate(models):
        model_df = df[df['model'] == model]

        # FP16 Batch 1
        batch1_fp16 = model_df[model_df['batch'] == 1]
        plot_subplot(axes[row_idx, 0], batch1_fp16, fp16_methods,
                     f'{model.upper()} - FP16 Batch 1',
                     show_y_label=True,
                     show_legend=False)

        # FP16 Batch 8
        batch8_fp16 = model_df[model_df['batch'] == 8]
        plot_subplot(axes[row_idx, 1], batch8_fp16, fp16_methods,
                     f'{model.upper()} - FP16 Batch 8',
                     show_y_label=False,
                     show_legend=False)

        # FP8 Batch 1
        batch1_fp8 = model_df[model_df['batch'] == 1]
        plot_subplot(axes[row_idx, 2], batch1_fp8, fp8_methods,
                     f'{model.upper()} - FP8 Batch 1',
                     show_y_label=False,
                     show_legend=False)

        # FP8 Batch 8
        batch8_fp8 = model_df[model_df['batch'] == 8]
        plot_subplot(axes[row_idx, 3], batch8_fp8, fp8_methods,
                     f'{model.upper()} - FP8 Batch 8',
                     show_y_label=False,
                     show_legend=False)

    # Add legend at the top center of the entire figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
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

    # Add common x-axis label
    fig.text(0.5, 0.02, 'Sequence Length', ha='center', fontsize=18)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.15, hspace=0.25)

    # Save figure
    plt.savefig('throughput_plot.pdf', format='pdf', bbox_inches='tight', dpi=300)
    print("Plot saved as throughput_plot.pdf")

if __name__ == '__main__':
    main() 
