import matplotlib.pyplot as plt
import numpy as np
import proc.behavior_metrics as bm


def plot_ami_across_days(results, animal, drop_n=3,
                         smooth=5, min_trials=10):
    """
    Plot trial-wise AMI across days for one animal.
    """

    days = sorted(results.get(animal, {}).keys())

    plt.figure(figsize=(8,5))

    n_plotted = 0

    for date in days:

        b = results[animal][date]

        trial_df = bm.build_trial_table(b, drop_n=drop_n)

        if len(trial_df) < min_trials:
            continue

        ami = trial_df["AMI_speed"].values

        # optional smoothing
        if smooth and len(ami) > smooth:
            kernel = np.ones(smooth)/smooth
            ami_plot = np.convolve(ami, kernel, mode="same")
        else:
            ami_plot = ami

        plt.plot(
            ami_plot,
            linewidth=1.5,
            alpha=0.8,
            label=date
        )

        n_plotted += 1

    plt.axhline(0, linestyle="--", color="k", alpha=0.5)

    plt.xlabel("Trial #")
    plt.ylabel("AMI (speed)")
    plt.title(f"{animal} — Trial-wise AMI across days")

    if n_plotted <= 10:
        plt.legend(fontsize=8)

    plt.tight_layout()
    plt.show()