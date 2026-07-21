import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def summarize_evacuation(evacuation_times):
    """
    Llogarit statistika bazë nga lista e kohëve të evakuimit
    (një vlerë për çdo boid - në cilin 'frame' ka dalë).
    """
    if len(evacuation_times) == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "std": None
        }

    times = np.array(evacuation_times)
    return {
        "count": len(times),
        "mean": np.mean(times),
        "median": np.median(times),
        "min": np.min(times),
        "max": np.max(times),
        "std": np.std(times)
    }


def print_summary(evacuation_times):
    """Shtyp në konsolë përmbledhjen e statistikave të evakuimit."""
    stats = summarize_evacuation(evacuation_times)

    print("\n--- Përmbledhja e Evakuimit ---")
    print(f"Numri i agjentëve të evakuuar: {stats['count']}")
    if stats['count'] > 0:
        print(f"Koha mesatare (mean):   {stats['mean']:.2f} frames")
        print(f"Koha mediane (median):  {stats['median']:.2f} frames")
        print(f"Koha minimale:          {stats['min']:.0f} frames")
        print(f"Koha maksimale:         {stats['max']:.0f} frames")
        print(f"Devijimi standard:      {stats['std']:.2f} frames")
    print("--------------------------------\n")


def plot_evacuation_histogram(evacuation_times, title="Shpërndarja e Kohës së Evakuimit"):
    """
    Vizaton një histogram që tregon sa agjentë kanë evakuuar
    në çdo interval kohor - kjo tregon vizualisht 'flow'-in e evakuimit
    (p.sh. nëse ka një 'bottleneck', shumë agjentë evakuojnë vonë).
    """
    plt.figure(figsize=(8, 5))
    plt.hist(evacuation_times, bins=20, color="#4a90d9", edgecolor="black")
    plt.xlabel("Koha e evakuimit (frames)")
    plt.ylabel("Numri i agjentëve")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_config_comparison(summary_csv_path="results/batch_results_summary.csv",
                            title="Krahasimi i Kohës së Evakuimit sipas Konfigurimit"):
    """
    Lexon CSV-në e përmbledhjes (nga batch_runner) dhe vizaton bar chart
    krahasues mes konfigurimeve, me error bars që tregojnë variabilitetin
    (std_of_means) mes trials të ndryshme.
    """
    df = pd.read_csv(summary_csv_path)

    plt.figure(figsize=(9, 6))
    bars = plt.bar(df["config"], df["mean_of_means"],
                    yerr=df["std_of_means"], capsize=8,
                    color="#4a90d9", edgecolor="black")

    # Shtojmë vlerën numerike mbi çdo shtyllë, për lexueshmëri më të mirë
    for bar, value in zip(bars, df["mean_of_means"]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                  f"{value:.1f}", ha="center", va="bottom", fontsize=10)

    plt.xlabel("Konfigurimi")
    plt.ylabel("Koha mesatare e evakuimit (frames)")
    plt.title(title)
    plt.tight_layout()
    plt.show()