import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.metrics import plot_config_comparison

plot_config_comparison(save_path="../results/config_comparison.png")