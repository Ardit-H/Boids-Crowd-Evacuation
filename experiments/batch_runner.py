import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from boids.simulation import Simulation
from analysis.metrics import summarize_evacuation, plot_config_comparison


def run_headless_simulation(num_boids, width, height, exits, obstacles=None, max_steps=2000):
    """
    Ekzekuton një simulim TË VETËM pa vizualizim (headless) - shumë
    më shpejt se me pygame, sepse s'humb kohë duke vizatuar.
    Kthen listën e kohëve të evakuimit të të gjithë agjentëve.
    """
    simulation = Simulation(num_boids, width, height, exits, obstacles)

    steps = 0
    while not simulation.is_finished() and steps < max_steps:
        simulation.step()
        steps += 1

    return simulation.evacuation_times


def run_batch(config_name, num_boids, width, height, exits, obstacles=None,
              num_trials=30, max_steps=2000):
    """
    Ekzekuton të njëjtin konfigurim 'num_trials' herë (me pozicione
    fillestare random të reja çdo herë), dhe mbledh statistikat
    e secilit trial - kjo eliminon 'fatin' e një ekzekutimi të vetëm.
    """
    print(f"\nDuke ekzekutuar konfigurimin: {config_name} ({num_trials} trials)...")

    results = []
    for trial in range(num_trials):
        evacuation_times = run_headless_simulation(
            num_boids, width, height, exits, obstacles, max_steps)
        stats = summarize_evacuation(evacuation_times)
        stats["config"] = config_name
        stats["trial"] = trial
        stats["num_exits"] = len(exits)
        results.append(stats)

        print(f"  Trial {trial + 1}/{num_trials} - mesatare: {stats['mean']:.1f} frames")

    return results


def main():
    WIDTH, HEIGHT = 900, 700
    NUM_BOIDS = 80
    NUM_TRIALS = 30

    # Konfigurimet që do t'i krahasojmë
    configs = {
        "1 derë - pa pengesë": ([(WIDTH / 2, 0)], []),
        "1 derë - me pengesë": ([(WIDTH / 2, 0)], [(WIDTH / 2 - 40, 150, 80, 40)]),
        "1 derë - pengesë kanalizuese": (
            [(WIDTH / 2, 0)],
            [
                (WIDTH / 2 - 150, 100, 60, 100),  # pengesë majtas rrugës
                (WIDTH / 2 + 90, 100, 60, 100),  # pengesë djathtas rrugës
            ]
        ),
        "2 dyer": ([(WIDTH / 4, 0), (3 * WIDTH / 4, 0)], []),
        "3 dyer": ([(WIDTH / 5, 0), (WIDTH / 2, 0), (4 * WIDTH / 5, 0)], []),
    }

    all_results = []
    for config_name, (exits, obstacles) in configs.items():
        results = run_batch(config_name, NUM_BOIDS, WIDTH, HEIGHT, exits,
                            obstacles=obstacles, num_trials=NUM_TRIALS)
        all_results.extend(results)

    # Konvertojmë në DataFrame (pandas) - lehtëson analizën dhe eksportimin
    df = pd.DataFrame(all_results)

    # Ruajmë rezultatet e plota (çdo trial individual) në CSV
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/batch_results_raw.csv", index=False)

    # Krijojmë dhe ruajmë përmbledhjen e krahasuar (mesatare e mesatareve, per config)
    summary = df.groupby("config").agg(
        mean_of_means=("mean", "mean"),
        std_of_means=("mean", "std"),
        mean_max=("max", "mean"),
        overall_std=("std", "mean")
    ).reset_index()

    summary.to_csv("results/batch_results_summary.csv", index=False)

    print("\n\n=== PËRMBLEDHJA FINALE (mesatare nga të gjitha trials) ===")
    print(summary.to_string(index=False))
    print(f"\nRezultatet u ruajtën në: results/batch_results_raw.csv")
    print(f"Përmbledhja u ruajt në: results/batch_results_summary.csv")

    plot_config_comparison()


if __name__ == "__main__":
    main()