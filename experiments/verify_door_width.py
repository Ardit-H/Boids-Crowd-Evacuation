# Test i dedikuar statistikor: a ndikon ndjeshëm gjerësia e derës
# (15px kundrejt 30px) në kohën e evakuimit? Përdor 80 trials (më
# shumë se 30 standarde te batch_runner.py) për siguri statistikore
# më të lartë, meqë synon një përgjigje specifike po/jo, jo thjesht
# krahasim i përgjithshëm konfigurimesh.

import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from experiments.batch_runner import run_batch
from analysis.metrics import plot_config_comparison

WIDTH, HEIGHT = 900, 700
NUM_TRIALS = 80  # shumë më shumë se 30, për siguri statistikore më të lartë.

configs = {
    "E ngushtë (15px)": ([("top", WIDTH / 2)], [], 15.0),
    "Mesatare (30px)": ([("top", WIDTH / 2)], [], 30.0),
}

all_results = []
for config_name, (exits, obstacles, exit_width) in configs.items():
    results = run_batch(config_name, 80, WIDTH, HEIGHT, exits,
                         obstacles=obstacles, exit_width=exit_width,
                         num_trials=NUM_TRIALS)
    all_results.extend(results)

df = pd.DataFrame(all_results)

results_dir = os.path.join(PROJECT_ROOT, "results")
os.makedirs(results_dir, exist_ok=True)

raw_path = os.path.join(results_dir, "door_width_verification_raw.csv")
summary_path = os.path.join(results_dir, "door_width_verification_summary.csv")

df.to_csv(raw_path, index=False)

summary = df.groupby("config").agg(
    mean_of_means=("mean", "mean"),
    std_of_means=("mean", "std"),
    mean_max=("max", "mean"),
    overall_std=("std", "mean")
).reset_index()

summary.to_csv(summary_path, index=False)

print("\n\n=== VERIFIKIM: 15px kundrejt 30px (80 trials secili) ===")
print(summary.to_string(index=False))

# Test i thjeshtë statistikor - a mbivendosen intervalet (mean ± std)?
# Rregull praktik (jo test formal si t-test): nëse diferenca mes
# mesatareve është më e vogël se 1 devijim standard, e trajtojmë si
# jo domethënëse - thjesht një kontroll i shpejtë, jo zëvendësim i
# një testi statistikor rigoroz.
row_15 = summary[summary["config"] == "E ngushtë (15px)"].iloc[0]
row_30 = summary[summary["config"] == "Mesatare (30px)"].iloc[0]

diff = abs(row_30["mean_of_means"] - row_15["mean_of_means"])
combined_std = (row_15["std_of_means"] + row_30["std_of_means"]) / 2

print(f"\nDiferenca mes mesatareve: {diff:.2f} frames")
print(f"Devijimi standard mesatar: {combined_std:.2f} frames")
print(f"Raporti diferencë/std: {diff / combined_std:.2f}")
if diff / combined_std < 1.0:
    print("=> Diferenca ka gjasa të MOS jetë domethënëse statistikisht (brenda 1 std).")
else:
    print("=> Diferenca ka gjasa të JETË domethënëse statistikisht (mbi 1 std).")

plot_config_comparison(summary_csv_path=summary_path,
                        title="Verifikim: 15px kundrejt 30px (80 trials)",
                        save_path=os.path.join(results_dir, "door_width_verification.png"))