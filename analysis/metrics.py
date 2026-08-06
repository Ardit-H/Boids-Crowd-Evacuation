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


def plot_evacuation_histogram(evacuation_times, title="Shpërndarja e Kohës së Evakuimit",
                                save_path=None, show=True):
    """
    Vizaton histogram dhe, nëse jepet save_path, e ruan si file .png.

    'show' kontrollon nëse thirret plt.show() (bllokues, hap event loop
    të vet GUI - TkAgg/QtAgg). VENDOSE show=False kur ky funksion
    thirret nga BRENDA një aplikacioni tjetër me event loop GUI (p.sh.
    pygame) - dy event loop GUI brenda TË NJËJTIT thread/proces (p.sh.
    Tkinter i matplotlib + SDL i pygame) shkaktojnë konflikte reale
    (dritare bosh, madhësi të prishura). Në atë rast, thirrësi duhet ta
    hapë vetë file-n e ruajtur (save_path) në një PROCES TË VEÇANTË -
    shih renderer.py, ku përdoret os.startfile/'open'/'xdg-open'.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(evacuation_times, bins=20, color="#4a90d9", edgecolor="black")
    plt.xlabel("Koha e evakuimit (frames)")
    plt.ylabel("Numri i agjentëve")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Grafiku u ruajt në: {save_path}")

    if show:
        plt.show()
    else:
        # Mbyll figurën eksplicitisht (jo plt.show()) - përndryshe
        # mbetet e hapur në memorie pafundësisht (memory leak i vogël,
        # por i grumbullueshëm nëse butoni klikohet shumë herë).
        plt.close()


def plot_config_comparison(summary_csv_path="results/batch_results_summary.csv",
                            title="Krahasimi i Kohës së Evakuimit sipas Konfigurimit",
                            save_path=None, show=True):
    """
    Lexon CSV-në e përmbledhjes dhe vizaton bar chart krahasues, plus
    e ruan si .png nëse jepet save_path. Shih shënimin te 'show' në
    plot_evacuation_histogram - vlen njësoj këtu.
    """
    df = pd.read_csv(summary_csv_path)

    plt.figure(figsize=(11, 7))
    bars = plt.bar(df["config"], df["mean_of_means"],
                    yerr=df["std_of_means"], capsize=8,
                    color="#4a90d9", edgecolor="black")

    for bar, value, std in zip(bars, df["mean_of_means"], df["std_of_means"]):
        # Numri vendoset saktë në mes të error bar-it (mesi mes
        # majës dhe fundit të tij, që përkon me vetë vlerën "value")
        plt.text(bar.get_x() + bar.get_width() / 2, value,
                 f"{value:.1f}", ha="center", va="center", fontsize=10,
                 bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

    plt.xlabel("Konfigurimi")
    plt.ylabel("Koha mesatare e evakuimit (frames)")
    plt.title(title)

    # Rrotullo emrat e konfigurimeve dhe zvogëlo pak fontin, që të mos
    # mbivendosen kur janë të gjatë (p.sh. "1 derë - pengesë kanalizuese")
    plt.xticks(rotation=15, ha="right", fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Grafiku u ruajt në: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()