import json
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from boids.polygon_simulation import PolygonSimulation
from analysis.metrics import summarize_evacuation


def run_headless_polygon(num_boids, vertices, doors, obstacles=None,
                          circle_obstacles=None, exit_width=15.0, max_steps=2000):
    """
    Ekzekuton një simulim TË VETËM (dhomë-poligon) pa vizualizim.
    """
    simulation = PolygonSimulation(num_boids, vertices, doors, obstacles,
                                    circle_obstacles, exit_width)
    steps = 0
    while not simulation.is_finished() and steps < max_steps:
        simulation.step()
        steps += 1
    return simulation.evacuation_times


def run_batch_polygon(config_name, num_boids, vertices, doors, obstacles=None,
                       circle_obstacles=None, exit_width=15.0, num_trials=30,
                       max_steps=2000):
    """Ekzekuton të njëjtin konfigurim poligon 'num_trials' herë."""
    print(f"\nDuke ekzekutuar (poligon - kontrolluar): {config_name} ({num_trials} trials)...")
    results = []
    for trial in range(num_trials):
        evacuation_times = run_headless_polygon(
            num_boids, vertices, doors, obstacles, circle_obstacles,
            exit_width, max_steps)
        stats = summarize_evacuation(evacuation_times)
        stats["config"] = config_name
        stats["trial"] = trial
        results.append(stats)
        print(f"  Trial {trial + 1}/{num_trials} - mesatare: {stats['mean']:.1f} frames")
    return results


def main():
    # PËRDOR configs_polygon_matched.json (sipërfaqe/distancë të barazuara mes
    # konfigurimeve të Grupit A, dhe distancë e barazuar mes Grupit B)
    configs_path = os.path.join(PROJECT_ROOT, "experiments", "configs_polygon_matched.json")
    with open(configs_path, "r", encoding="utf-8") as f:
        configs = json.load(f)

    all_results = []
    for config_name, cfg in configs.items():
        results = run_batch_polygon(
            config_name,
            cfg["num_boids"],
            cfg["vertices"],
            cfg["doors"],
            obstacles=cfg.get("obstacles", []),
            circle_obstacles=cfg.get("circle_obstacles", []),
            exit_width=cfg.get("exit_width", 15.0),
            num_trials=30
        )
        all_results.extend(results)

    df = pd.DataFrame(all_results)

    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    df.to_csv(os.path.join(results_dir, "batch_results_polygon_matched_raw.csv"), index=False)

    summary = df.groupby("config").agg(
        mean_of_means=("mean", "mean"),
        std_of_means=("mean", "std"),
        mean_max=("max", "mean"),
        overall_std=("std", "mean")
    ).reset_index()
    summary.to_csv(os.path.join(results_dir, "batch_results_polygon_matched_summary.csv"), index=False)

    print("\n\n=== PËRMBLEDHJA (POLIGON - KONTROLLUAR: sip./dist. të barazuara) ===")
    print(summary.to_string(index=False))
    print(f"\nRezultatet u ruajtën në: results/batch_results_polygon_matched_raw.csv")
    print(f"Përmbledhja u ruajt në: results/batch_results_polygon_matched_summary.csv")


if __name__ == "__main__":
    main()