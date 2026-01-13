from collections import defaultdict
import glob
import os
import re

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

def custom_colormap():
    """Create custom colormap from hex colors."""
    hex_colors = ['#EADFB4','#9BB0C1','#F6995C','#874C62']
    rgb = [mpl.colors.to_rgb(c) for c in hex_colors]
    return mpl.colors.LinearSegmentedColormap.from_list('custom_cmap', rgb)

def extract_models_from_spawner(spawner_file='benchmark_jobs_spawner.sh'):
    """Extract model names and frameworks from spawner script."""
    models, frameworks = [], []
    try:
        with open(spawner_file, 'r') as f:
            content = f.read()

        # Extract models and frameworks using regex
        models_match = re.search(r'declare -A models=\((.*?)\)', content, re.DOTALL)
        frameworks_match = re.search(r'all_frameworks=\((.*?)\)', content, re.DOTALL)

        if models_match:
            models = re.findall(r'\["([^"]+)"\]', models_match.group(1))
        if frameworks_match:
            frameworks = re.findall(r'"([^"]+)"', frameworks_match.group(1))

    except Exception as e:
        print(f"Error reading spawner file: {e}")
        # Fallback defaults
        models = ["facebook/opt-1.3b"]
        frameworks = ["deepspeed", "hf_accelerate", "llmlingua", "vllm", "vllm_async"]

    return models, frameworks

def find_csv_file(model_dir, framework, input_size, output_size):
    """Find CSV file for given parameters."""
    output_dir = os.path.join(model_dir, framework, str(input_size), str(output_size))
    if not os.path.exists(output_dir):
        return None

    # Look for timestamped results first, then fallback
    results_dirs = glob.glob(os.path.join(output_dir, "results_*"))
    if results_dirs:
        latest_dir = max(results_dirs, key=os.path.getmtime)
        csv_file = os.path.join(latest_dir, "benchmark_summary.csv")
    else:
        csv_file = os.path.join(output_dir, "results", "benchmark_summary.csv")

    return csv_file if os.path.exists(csv_file) else None

def extract_performance_value(csv_file, expected_framework):
    """Extract performance value from CSV file."""
    try:
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            if row['framework'] == expected_framework:
                return float(row['token_per_sec_avg'])
    except Exception:
        pass
    return None

def process_model(results_dir, model_name, frameworks, output_dir):
    """Load data and create heatmap for a single model."""
    sizes = [2**i for i in range(10)]
    model_dir = os.path.join(results_dir, model_name.replace('/', '_'))

    if not os.path.exists(model_dir):
        print("  Model directory not found, skipping")
        return

    # Load data for this model
    data = defaultdict(dict)
    stats = defaultdict(int)

    for framework in frameworks:
        for input_size in sizes:
            for output_size in sizes:
                csv_path = find_csv_file(model_dir, framework, input_size, output_size)
                if csv_path:
                    value = extract_performance_value(csv_path, framework)
                    if value:
                        data[framework][(input_size, output_size)] = value
                        stats[framework] += 1

    if not any(stats.values()):
        print("  No data found, skipping")
        return

    # Print summary
    summary = ", ".join([f"{fw}: {stats[fw]} points" for fw in frameworks if stats[fw] > 0])
    print(f"  Data summary: {summary}")

    # Get available frameworks with data
    available_frameworks = [fw for fw in frameworks if fw in data and data[fw]]
    if not available_frameworks:
        print("  No valid data for visualization, skipped")
        return

    # Calculate value range for color scaling
    all_values = [
        data[fw][(i, o)]
        for fw in available_frameworks
        for i in sizes for o in sizes
        if (i, o) in data[fw]
    ]

    if not all_values:
        print("  No valid data for visualization, skipped")
        return

    vmin, vmax = min(all_values), max(all_values)

    # Create figure
    rows, cols = 2, 3
    base_size = 0.225
    fig_width = cols * (base_size * len(sizes) + 0.8) + 1.2
    fig_height = rows * (base_size * len(sizes) + 0.8) + 0.45

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(rows, cols, left=0.06, right=0.89, top=0.92, bottom=0.08, hspace=0.15, wspace=0.15)
    cmap = custom_colormap()

    # Create subplots for each framework
    for idx, framework in enumerate(frameworks):
        if idx >= rows * cols:
            break

        row_idx, col_idx = idx // cols, idx % cols
        ax = fig.add_subplot(gs[row_idx, col_idx])

        # Prepare heatmap data
        heatmap_data = np.full((len(sizes), len(sizes)), np.nan)
        if framework in available_frameworks:
            for i_idx, i in enumerate(sizes):
                for o_idx, o in enumerate(sizes):
                    if (i, o) in data[framework]:
                        heatmap_data[i_idx, o_idx] = data[framework][(i, o)]

        # Create annotations
        annotations = np.array([
            ["N/A" if np.isnan(val) else f"{int(round(val))}" for val in row]
            for row in heatmap_data
        ])

        # Create heatmap
        mask = np.isnan(heatmap_data)
        sns.heatmap(heatmap_data, annot=annotations, fmt="", cmap=cmap,
                   xticklabels=sizes, yticklabels=sizes, mask=mask, ax=ax,
                   vmin=vmin, vmax=vmax, cbar=False, annot_kws={"size": 7})

        # Style subplot
        ax.set_title(framework, fontsize=10, pad=2)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis='both', labelsize=7)
        if len(sizes) > 5:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Hide empty subplots
    for idx in range(len(frameworks), rows * cols):
        row_idx, col_idx = idx // cols, idx % cols
        if row_idx < rows and col_idx < cols:
            fig.add_subplot(gs[row_idx, col_idx]).set_visible(False)

    # Add titles and labels
    model_display = model_name.replace('/', '_')
    fig.suptitle(f"{model_display} Performance Comparison (tokens/sec)", fontsize=12, y=0.98)
    fig.text(0.5, 0.01, 'Output Tokens', ha='center', fontsize=11)
    fig.text(0.01, 0.5, 'Input Tokens', va='center', rotation='vertical', fontsize=11)

    # Add colorbar
    cax = fig.add_axes([0.91, 0.1, 0.015, 0.8])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, label="Tokens per second")
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_yticks(np.linspace(vmin, vmax, 8))
    cbar.ax.set_yticklabels([f"{int(round(x))}" for x in np.linspace(vmin, vmax, 8)])

    # Save and close
    safe_name = model_name.replace('/', '_')
    fig.savefig(f"{output_dir}/{safe_name}_combined_heatmap.png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("  Saved heatmap")

def create_heatmaps(software_setup='cuda121_torch212'):
    """Create heatmaps for specified software setup."""
    results_dir = f'experiments_{software_setup}'

    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} does not exist")
        return

    # Load configuration
    models, frameworks = extract_models_from_spawner()
    print(f"Found {len(models)} models and {len(frameworks)} frameworks")

    # Create output directory
    output_dir = f"heatmaps_{software_setup}"
    os.makedirs(output_dir, exist_ok=True)

    # Process each model
    for model_name in models:
        print(f"Processing model: {model_name}")
        process_model(results_dir, model_name, frameworks, output_dir)

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python heatmaps.py <software_setup>")
        print("  software_setup: cuda121_torch212|cuda126_torch260")
        sys.exit(1)

    software_setup = sys.argv[1]
    if software_setup not in ['cuda121_torch212', 'cuda126_torch260']:
        print("Error: Invalid software_setup. Use 'cuda121_torch212' or 'cuda126_torch260'")
        sys.exit(1)

    print(f"Generating heatmaps for {software_setup} software setup")
    create_heatmaps(software_setup)
    print("Heatmap generation complete")
