import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (same as plot_from_csv.py)
import matplotlib.pyplot as plt
import textwrap

# Load data
df = pd.read_csv('ablation.csv')

# Data processing
categories = df.columns[1:]
values = df.iloc[0, 1:].values
benchmark_name = df.iloc[0, 0]

# Wrap labels for the x-axis to avoid overlapping
wrapped_labels = ['\n'.join(textwrap.wrap(l, 15)) for l in categories]

# Plotting
# Increased figure size slightly (12x8) to better accommodate larger fonts
fig, ax = plt.subplots(figsize=(12, 8))
# Updated colors: keep green for full performance, use warm gradient for others
bars = ax.bar(categories, values, tick_label=wrapped_labels, color=['#2ca02c', '#ffa500', '#ff6b6b', '#c084fc'])

# Styling with larger fonts
ax.set_ylabel('Latency', fontsize=24)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

# Increase tick label sizes
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)

# Adjust layout to prevent clipping
plt.tight_layout()

# Save the figure
plt.savefig('ablation_study.pdf')