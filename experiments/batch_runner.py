import json
import numpy as np
import pandas as pd
import sys
import os



# Përcakton rrënjën e projektit bazuar te vendndodhja e këtij file-i,
# jo te "working directory" e ekzekutimit - kështu funksionon njësoj
# pavarësisht se si/nga ku ekzekutohet skripti.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
from boids.simulation import Simulation
from analysis.metrics import summarize_evacuation, plot_config_comparison


def run_headless_simulation(num_boids, width, height, exits, obstacles=None,
                             exit_width=15.0, max_steps=2000):
    """
    Ekzekuton një simulim TË VETËM pa vizualizim (headless). Kthen
    listën e kohëve të evakuimit (frame) për çdo boid që doli, ose
    e ndërpret te max_steps nëse disa boid mbeten të ngërçuar
    pafundësisht (mbrojtje kundër loop-eve të pafund në batch runs).
    """
    simulation = Simulation(num_boids, width, height, exits, obstacles, exit_width)

    steps = 0
    while not simulation.is_finished() and steps < max_steps:
        simulation.step()
        steps += 1

    return simulation.evacuation_times


def run_batch(config_name, num_boids, width, height, exits, obstacles=None,
              exit_width=15.0, num_trials=30, max_steps=2000):
    """
    Ekzekuton të njëjtin konfigurim 'num_trials' herë (me pozicione
    fillestare random të reja çdo herë), dhe mbledh statistikat
    e secilit trial - kjo eliminon 'fatin' e një ekzekutimi të vetëm.
    """
    print(f"\nDuke ekzekutuar konfigurimin: {config_name} ({num_trials} trials)...")

    results = []
    for trial in range(num_trials):
        evacuation_times = run_headless_simulation(
            num_boids, width, height, exits, obstacles, exit_width, max_steps)
        stats = summarize_evacuation(evacuation_times)
        stats["config"] = config_name
        stats["trial"] = trial
        stats["num_exits"] = len(exits)
        results.append(stats)

        # Print progresi për çdo trial - batch runs mund të zgjasin
        # disa minuta, kështu përdoruesi sheh që programi ende punon.
        print(f"  Trial {trial + 1}/{num_trials} - mesatare: {stats['mean']:.1f} frames")

    return results


def main():
    WIDTH, HEIGHT = 900, 700
    NUM_TRIALS = 30

    # Konfigurimet e eksperimenteve përcaktohen në experiments/configs.json
    # (jo hard-coded këtu), që të mund të shtohen/ndryshohen skenarë të
    # rinj pa prekur kodin.
    # Struktura e configs.json: {emri_konfigurimit: {num_boids, exits,
    # obstacles (opsionale), exit_width (opsionale, default 15.0)}}.
    # "exits" është listë [x, 0] (formati i vjetër, gjithmonë muri i
    # sipërm) ose [side, pos] (formati i ri me 4 mure - shih Environment).
    configs_path = os.path.join(PROJECT_ROOT, "experiments", "configs.json")
    with open(configs_path, "r", encoding="utf-8") as f:
        configs = json.load(f)

    all_results = []
    for config_name, cfg in configs.items():
        results = run_batch(
            config_name,
            cfg["num_boids"], WIDTH, HEIGHT,
            cfg["exits"],
            obstacles=cfg.get("obstacles", []),
            exit_width=cfg.get("exit_width", 15.0),
            num_trials=NUM_TRIALS
        )
        all_results.extend(results)

    df = pd.DataFrame(all_results)

    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    df.to_csv(os.path.join(results_dir, "batch_results_raw.csv"), index=False)

    # Grupon rezultatet e 30 trials-ave për konfigurim, duke llogaritur
    # mesataren e mesatareve (jo vetëm një ekzekutim të vetëm) - kjo
    # eliminon "fatin" e një ekzekutimi random të vetëm.
    summary = df.groupby("config").agg(
        mean_of_means=("mean", "mean"),
        std_of_means=("mean", "std"),
        mean_max=("max", "mean"),
        overall_std=("std", "mean")
    ).reset_index()

    summary.to_csv(os.path.join(results_dir, "batch_results_summary.csv"), index=False)

    print("\n\n=== PËRMBLEDHJA FINALE (mesatare nga të gjitha trials) ===")
    print(summary.to_string(index=False))
    print(f"\nRezultatet u ruajtën në: results/batch_results_raw.csv")
    print(f"Përmbledhja u ruajt në: results/batch_results_summary.csv")

    summary_csv_path = os.path.join(results_dir, "batch_results_summary.csv")
    plot_config_comparison(summary_csv_path=summary_csv_path,
                           save_path=os.path.join(results_dir, "config_comparison.png"))


if __name__ == "__main__":
    main()