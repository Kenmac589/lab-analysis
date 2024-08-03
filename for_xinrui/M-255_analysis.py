import dlc2kinematics as dlck
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dlc2kinematics import Visualizer2D
from kinsynpy import dlctools as dlt


def step_cycle_trial(
    input_h5, fig_filename, fig_title, cycle_filename, save_plots=True
):
    """Extracts step cycles from h5 output by deeplabcut

    Parameters
    ----------
    input_h5:
        The h5 data file that is automatically produced when analyzing videos.
    fig_filename:
        Where you want the figure saved to as well as the name of the file.
    cycle_filename:
        Where you want the csv output containing the cycle timings to be saved.
    save_plots:
        Whether or not you want to save plots and cycles default set to true.

    Returns
    -------
    None:

    """

    # Loading in a dataset
    df, bodyparts, scorer = dlck.load_data(input_h5)

    # Grabbing toe marker data
    toe = df[scorer]["toe"]

    # Converting to numpy array
    toe_np = pd.array(toe["x"])

    # Filtering to clean up traces like you would in spike
    toe_filtered = dlt.median_filter(toe_np, 9)
    # toe_roi_selection = toe_np[0:2550]  # Just to compare to original

    # Cleaning up selection to region before mouse moves back
    # toe_roi_selection_fil = toe_filtered[0:2550]

    # Calling function for swing estimation
    swing_onset, swing_offset = dlt.swing_estimation(toe_filtered)

    step_cyc_durations = dlt.step_cycle_est(toe_filtered)

    # Saving values
    np.savetxt(cycle_filename, step_cyc_durations, delimiter=",")

    # Calling function for step cycle calculation

    # Some of my default plotting parameters I like
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set(style="white", font_scale=1.5, rc=custom_params)

    # Plot Legend
    swing_legend = [
        "Limb X cord",
        "Swing offset",
        "Swing onset",
    ]

    fig, axs = plt.subplots(2)
    fig.suptitle(fig_title)

    # For plotting figure demonstrating how swing estimation was done
    axs[0].set_title("Swing Estimation")
    axs[0].plot(toe_filtered)
    axs[0].plot(swing_offset, toe_filtered[swing_offset], "^")
    axs[0].plot(swing_onset, toe_filtered[swing_onset], "v")
    axs[0].legend(swing_legend, loc="best")

    # Showing results for step cycle timing
    axs[1].set_title("Step Cycle Result")
    axs[1].bar(
        0, np.mean(step_cyc_durations), yerr=np.std(step_cyc_durations), capsize=5
    )

    # Saving Figure in same folder
    fig = mpl.pyplot.gcf()
    fig.set_size_inches(19.8, 10.80)

    if save_plots is True:
        # Saving values
        np.savetxt(cycle_filename, step_cyc_durations, delimiter=",")
        plt.savefig(fig_filename, dpi=300)
        print("Results saved")
    else:
        print("Results not saved.")
        plt.show()


def main():

    step_cycle_trial(
        "./xinrui_M-255/2024-06-13_000000DLC_resnet50_dtr_update_predtxApr8shuffle1_1110000_filtered.h5",
        fig_filename="./test_kinematics-2024-06-13_000000.png",
        fig_title="Step Cycle for Video 2024-06-13_000000",
        cycle_filename="./step_cycles-2024-06-13_000000.csv",
        save_plots=True,
    )

    step_cycle_trial(
        "./xinrui_M-255/ts3_000000DLC_resnet50_dtr_update_predtxApr8shuffle1_1110000_filtered.h5",
        fig_filename="./test_kinematics-ts3_000000.png",
        fig_title="Step Cycle for Video ts_000000",
        cycle_filename="./step_cycles-ts3_000000.csv",
        save_plots=True,
    )

    step_cycle_trial(
        "./xinrui_M-255/ts3_000001DLC_resnet50_dtr_update_predtxApr8shuffle1_1110000.h5",
        fig_filename="./test_kinematics-ts3_000001.png",
        fig_title="Step Cycle for Video ts_000001",
        cycle_filename="./step_cycles-ts3_000001.csv",
        save_plots=True,
    )


if __name__ == "__main__":
    main()
