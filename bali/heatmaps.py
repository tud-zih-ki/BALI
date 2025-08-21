import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import glob
from collections import defaultdict


def custom_colormap():
    """Create custom colormap from hex colors."""
    hex_colors = ['#EADFB4', '#9BB0C1', '#F6995C', '#874C62']
    rgb = [mpl.colors.to_rgb(c) for c in hex_colors]
    return mpl.colors.LinearSegmentedColormap.from_list('custom_cmap', rgb)


def find_latest_results_dir():
    """Find the most recent bali_results_* directory."""
    results_dirs = glob.glob("bali_results_*")
    return max(results_dirs, key=os.path.getmtime) if results_dirs else None


def extract_performance_value(csv_file, framework):
    """Extract performance value from CSV file."""
    try:
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            if row['framework'] == framework:
                return float(row['token_per_sec_avg'])
    except Exception:
        pass
    return None


def collect_data(results_dir, model_filter=None, framework_filter=None):
    """Collect performance data from hierarchical results directory structure."""
    data = defaultdict(lambda: defaultdict(dict))
    available_models, available_frameworks = set(), set()
    
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} does not exist")
        return data, available_models, available_frameworks
    
    # Traverse: model/framework/input_len/output_len/
    for model_name in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model_name)
        if not os.path.isdir(model_path) or (model_filter and model_name not in model_filter):
            continue
        
        for framework in os.listdir(model_path):
            framework_path = os.path.join(model_path, framework)
            if not os.path.isdir(framework_path) or (framework_filter and framework not in framework_filter):
                continue
            
            for input_len_str in os.listdir(framework_path):
                input_len_path = os.path.join(framework_path, input_len_str)
                if not os.path.isdir(input_len_path):
                    continue
                
                try:
                    input_len = int(input_len_str)
                except ValueError:
                    continue
                
                for output_len_str in os.listdir(input_len_path):
                    output_len_path = os.path.join(input_len_path, output_len_str)
                    if not os.path.isdir(output_len_path):
                        continue
                    
                    try:
                        output_len = int(output_len_str)
                    except ValueError:
                        continue
                    
                    csv_file = os.path.join(output_len_path, "benchmark_summary.csv")
                    if not os.path.exists(csv_file):
                        continue
                    
                    value = extract_performance_value(csv_file, framework)
                    if value is not None:
                        data[model_name][framework][(input_len, output_len)] = value
                        available_models.add(model_name)
                        available_frameworks.add(framework)
    
    return data, available_models, available_frameworks


def create_model_heatmap(model_data, model_name, frameworks, output_dir):
    """Create heatmap for a single model across multiple frameworks."""
    if not model_data:
        print(f"  No data found for model {model_name}, skipping")
        return
    
    # Get all input/output sizes and available frameworks
    all_sizes = set()
    available_frameworks = []
    for fw in frameworks:
        if fw in model_data and model_data[fw]:
            available_frameworks.append(fw)
            for (input_len, output_len) in model_data[fw].keys():
                all_sizes.update([input_len, output_len])
    
    if not available_frameworks or not all_sizes:
        print(f"  No valid data for model {model_name}, skipped")
        return
    
    sizes = sorted(all_sizes)
    
    # Get value range for color scaling
    all_values = [
        model_data[fw][(i, o)] 
        for fw in available_frameworks 
        for i in sizes for o in sizes 
        if (i, o) in model_data[fw]
    ]
    
    if not all_values:
        print(f"  No valid values for model {model_name}, skipped")
        return
    
    vmin, vmax = min(all_values), max(all_values)
    
    # Simple layout calculation - let matplotlib handle the rest
    n_frameworks = len(available_frameworks)
    cols = min(3, n_frameworks)
    rows = (n_frameworks + cols - 1) // cols
    
    # Calculate figure size based on fixed cell size and number of frameworks
    cell_size = 0.8  # Fixed size per heatmap cell in inches
    heatmap_width = len(sizes) * cell_size + 2.0  # Add space for labels
    heatmap_height = len(sizes) * cell_size + 1.5  # Add space for labels
    
    # Total figure size based on subplot layout
    fig_width = cols * heatmap_width + 2.0  # Add space between subplots and colorbar
    fig_height = rows * heatmap_height + 1.5  # Add space for title and between subplots
    
    # Use constrained_layout for automatic spacing
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), 
                            constrained_layout=True)
    
    # Handle single subplot case
    if n_frameworks == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    else:
        axes = axes.flatten()
    
    cmap = custom_colormap()
    
    # Create subplots for each framework
    for idx, framework in enumerate(available_frameworks):
        ax = axes[idx]
        
        # Prepare heatmap data
        heatmap_data = np.full((len(sizes), len(sizes)), np.nan)
        for i_idx, i in enumerate(sizes):
            for o_idx, o in enumerate(sizes):
                if (i, o) in model_data[framework]:
                    heatmap_data[i_idx, o_idx] = model_data[framework][(i, o)]
        
        # Create annotations
        annotations = [
            ["N/A" if np.isnan(val) else f"{int(round(val))}" for val in row]
            for row in heatmap_data
        ]
        
        # Create heatmap with shared colorbar
        mask = np.isnan(heatmap_data)
        sns.heatmap(heatmap_data, annot=annotations, fmt="", cmap=cmap,
                   xticklabels=sizes, yticklabels=sizes, mask=mask, ax=ax,
                   vmin=vmin, vmax=vmax, cbar=(idx == 0),  # Only first plot gets colorbar
                   annot_kws={"size": 10},
                   linewidths=0.5, linecolor='white')
        
        # Simple, clean styling
        ax.set_title(framework, fontsize=14, pad=20)
        ax.set_xlabel("Output Tokens", fontsize=12)
        ax.set_ylabel("Input Tokens", fontsize=12)
        ax.tick_params(axis='both', labelsize=10)
        
        # Rotate labels if needed
        if len(sizes) > 5:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    # Hide unused subplots
    for idx in range(len(available_frameworks), len(axes)):
        axes[idx].set_visible(False)
    
    # Simple title
    fig.suptitle(f"{model_name} Performance (tokens/sec)", 
                fontsize=16)
    
    # Save with automatic tight layout
    safe_name = model_name.replace('/', '_')
    fig.savefig(f"{output_dir}/{safe_name}_heatmap.png", 
               bbox_inches='tight', dpi=150, facecolor='white')
    print(f"  Saved heatmap for {model_name}")
    plt.show()



def create_heatmaps(results_dir=None, model_names=None, frameworks=None, output_dir=None):
    """
    Create heatmaps from BALI results.
    """
    # Use latest results directory if not specified
    if results_dir is None:
        results_dir = find_latest_results_dir()
        if results_dir is None:
            print("No results directories found (bali_results_*)")
            return
        print(f"Using latest results directory: {results_dir}")
    
    # Set default output directory
    if output_dir is None:
        output_dir = os.path.join(results_dir, "heatmaps")
    
    # Collect data
    print("Collecting data from results...")
    data, available_models, available_frameworks = collect_data(results_dir, model_names, frameworks)
    
    if not available_models:
        print("No data found in results directory")
        return
    
    # Filter to available data
    target_models = list(available_models) if model_names is None else [m for m in model_names if m in available_models]
    target_frameworks = list(available_frameworks) if frameworks is None else [f for f in frameworks if f in available_frameworks]
    
    if not target_models or not target_frameworks:
        print("No valid models or frameworks found")
        return
    
    print(f"Found data for {len(available_models)} models and {len(available_frameworks)} frameworks")
    print(f"Will plot {len(target_models)} models: {target_models}")
    print(f"Will plot {len(target_frameworks)} frameworks: {target_frameworks}")
    
    # Create heatmaps
    os.makedirs(output_dir, exist_ok=True)
    for model_name in target_models:
        print(f"Processing model: {model_name}")
        if model_name in data:
            create_model_heatmap(data[model_name], model_name, target_frameworks, output_dir)
        else:
            print(f"  No data found for {model_name}")
    
    print(f"Heatmaps saved to: {output_dir}")