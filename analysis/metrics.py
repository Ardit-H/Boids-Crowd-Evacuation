import numpy as np
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