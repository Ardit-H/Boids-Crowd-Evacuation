import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.metrics import plot_config_comparison

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
results_dir = os.path.join(PROJECT_ROOT, "results")

plot_config_comparison(
    summary_csv_path=os.path.join(results_dir, "batch_results_summary.csv"),
    save_path=os.path.join(results_dir, "config_comparison.png")
)