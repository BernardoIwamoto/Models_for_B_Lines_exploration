from pathlib import Path

import matplotlib.pyplot as plt


def plot_curves(curves, title, ylabel, save_path):

    plt.figure(figsize=(10,5))

    for x, y, label in curves:

        plt.plot(x, y, label=label)

    plt.title(title)

    plt.xlabel("Iteration")

    plt.ylabel(ylabel)

    plt.grid(True)

    plt.legend()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()