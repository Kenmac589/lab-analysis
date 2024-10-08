# import dlc2kinematics as dlck
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from matplotlib.lines import Line2D

# from dlc2kinematics import Visualizer2D
# from kinsynpy import dlctools as dlt
# from kinsynpy import latstability as ls


def stance_only(swon_times, swoff_times):
    for i in range(len(swon_times)):
        counter = 0
        cur_swoff = 0

        cur_swon = swon_times[i]
        cur_swoff = swoff_times[cur_swoff]

        while cur_swoff < cur_swon:

            cur_swoff = swoff_times[counter]

            counter = counter + 1

        print(f"SWON {cur_swon}")
        print(f"SWOFF {cur_swoff}\n")


def combine_arrays(arrays):

    max_length = max([len(a) for a in arrays])

    combined = np.zeros((len(arrays), max_length))

    for i, arr in enumerate(arrays):
        combined[i, : len(arr)] = arr

    return combined


def plot_joint(
    input_dataframe,
    swon_ch,
    swoff_ch,
    joint_ch,
    joint_name,
    condition_name,
    filename="./joint_test.svg",
    save=False,
):

    event_marker = 1

    kindf_subset = input_dataframe.loc[:, ["Time", swon_ch, swoff_ch, joint_ch]]

    # Getting timings
    # kindf_subset = kindf_subset.set_index("Time")

    # time_value = kindf_subset["Time"].to_numpy(dtype=float)
    joint_value = kindf_subset[joint_ch].to_numpy(dtype=float)

    swon_times = kindf_subset.loc[kindf_subset[swon_ch] == event_marker].index.tolist()
    swoff_times = kindf_subset.loc[
        kindf_subset[swoff_ch] == event_marker
    ].index.tolist()

    # Plotting settings
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set(
        style="white",
        # font="serif",
        font_scale=1.2,
        palette="colorblind",
        rc=custom_params,
    )
    plt.title(f"{joint_name} Joint Angle for {condition_name}")

    # hip_all_steps = np.array([])
    for i in range(len(swon_times) - 1):

        if swon_times[i] < swoff_times[-1]:
            step_begin = swon_times[i]
            step_end = swon_times[i + 1]
            joint_step = joint_value[step_begin:step_end]
            joint_step = sp.signal.savgol_filter(joint_step, 30, 3)
            if len(joint_step) < 800:
                # hip_all_steps = np.vstack((hip_all_steps, joint_step))
                plt.plot(joint_step, color="black", alpha=0.2)
        # hip_angle = np.array([])

        # hip_angle = np.append(hip_angle, [[]])

    fig = mpl.pyplot.gcf()
    fig.set_size_inches(15.8, 10.80)
    fig.tight_layout()

    if save is True:
        plt.savefig(filename, dpi=300)
        plt.show()
    else:
        plt.show()


def joint_fig(
    ax,
    pre_df,
    post_df,
    dtx_df,
    swon_ch,
    swoff_ch,
    joint_ch,
    joint_name,
    condition="EMG-test",
):

    event_marker = 1

    predf_subset = pre_df.loc[:, ["Time", swon_ch, swoff_ch, joint_ch]]
    postdf_subset = post_df.loc[:, ["Time", swon_ch, swoff_ch, joint_ch]]
    dtx_subset = dtx_df.loc[:, ["Time", swon_ch, swoff_ch, joint_ch]]

    pre_joint_value = predf_subset[joint_ch].to_numpy(dtype=float)
    post_joint_value = postdf_subset[joint_ch].to_numpy(dtype=float)
    dtx_joint_value = dtx_subset[joint_ch].to_numpy(dtype=float)

    # Getting timings

    # Pre
    pre_swon_times = predf_subset.loc[
        predf_subset[swon_ch] == event_marker
    ].index.tolist()
    pre_swoff_times = predf_subset.loc[
        predf_subset[swoff_ch] == event_marker
    ].index.tolist()

    # Post
    post_swon_times = postdf_subset.loc[
        postdf_subset[swon_ch] == event_marker
    ].index.tolist()
    post_swoff_times = postdf_subset.loc[
        postdf_subset[swoff_ch] == event_marker
    ].index.tolist()

    # DTX
    dtx_swon_times = dtx_subset.loc[dtx_subset[swon_ch] == event_marker].index.tolist()
    dtx_swoff_times = dtx_subset.loc[
        dtx_subset[swoff_ch] == event_marker
    ].index.tolist()

    ax.set_ylabel(f"{joint_name}")

    legend_list = ["Pre-EMG", "Post-EMG", "Post-DTX"]
    for i in range(len(pre_swon_times) - 1):

        # Exclusion of prolonged periods indicating time between recordings
        if pre_swon_times[i] < pre_swoff_times[-1]:
            step_begin = pre_swon_times[i]
            step_end = pre_swon_times[i + 1]
            joint_step = pre_joint_value[step_begin:step_end]
            joint_step = sp.signal.savgol_filter(joint_step, 30, 3)
            if len(joint_step) < 800:
                ax.plot(joint_step, color="black", alpha=0.2, label="Pre-EMG")

    for i in range(len(post_swon_times) - 1):

        # Exclusion of prolonged periods indicating time between recordings
        if post_swon_times[i] < post_swoff_times[-1]:
            step_begin = post_swon_times[i]
            step_end = post_swon_times[i + 1]
            joint_step = post_joint_value[step_begin:step_end]
            joint_step = sp.signal.savgol_filter(joint_step, 30, 3)
            if len(joint_step) < 800:
                ax.plot(joint_step, color="red", alpha=0.2, label="post-EMG")

    for i in range(len(dtx_swon_times) - 1):

        # Exclusion of prolonged periods indicating time between recordings
        if dtx_swon_times[i] < dtx_swoff_times[-1]:
            step_begin = dtx_swon_times[i]
            step_end = dtx_swon_times[i + 1]
            joint_step = dtx_joint_value[step_begin:step_end]
            joint_step = sp.signal.savgol_filter(joint_step, 30, 3)
            if len(joint_step) < 800:
                ax.plot(joint_step, color="blue", alpha=0.2, label="dtx-EMG")

    custom_lines = [
        Line2D([0], [0], color="black", lw=4),
        Line2D([0], [0], color="red", lw=4),
        Line2D([0], [0], color="blue", lw=4),
    ]
    ax.legend(custom_lines, legend_list, loc="best")
    out = mpl.pyplot.gcf()

    # if save_fig is True:
    #     plt.savefig(f"./joint_angles/{condition}-{joint_name}.svg")

    return out


def main():

    # NOTE: At this time I am using videos from:
    # - M1-pre-emg: vids 3, 4
    # - M2-pre-emg: vids 0, 1, 2
    # - M1-post-emg: vids 0, 1, 2, 3
    # - M2-post-emg: vids 1, 8, 9

    # predf = pd.read_csv("./EMG-test-1/emg-test-1-pre-emg.txt", delimiter=",", header=0)
    # postdf = pd.read_csv(
    #     "./EMG-test-1/emg-test-1-post-emg.txt", delimiter=",", header=0
    # )

    predf = pd.read_csv("./EMG-test-2/emg-test-2-pre-emg.txt", delimiter=",", header=0)
    postdf = pd.read_csv(
        "./EMG-test-2/emg-test-2-post-emg.txt", delimiter=",", header=0
    )
    dtxdf = pd.read_csv("./EMG-test-2/emg-test-2-post-dtx.txt", delimiter=",", header=0)

    condition = "EMG-test-2"
    save_plots = False
    figure_filename = "./joint_angles/emg-test-2-with-dtx.pdf"

    # Plotting settings
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set(
        style="white",
        # font="serif",
        font_scale=1.0,
        palette="colorblind",
        rc=custom_params,
    )

    fig, (ax1, ax2, ax3) = plt.subplots(3)

    fig.suptitle(condition)

    # Multi thing test
    joint_fig(
        ax1,
        predf,
        postdf,
        dtxdf,
        swon_ch="31 swon",
        swoff_ch="32 swoff",
        joint_ch="15 hip ang",
        joint_name="Hip",
        condition=condition,
    )

    # For Knee
    joint_fig(
        ax2,
        predf,
        postdf,
        dtxdf,
        swon_ch="31 swon",
        swoff_ch="32 swoff",
        joint_ch="16 knee ang",
        joint_name="Knee",
        condition=condition,
    )

    # For Ankle
    joint_fig(
        ax3,
        predf,
        postdf,
        dtxdf,
        swon_ch="31 swon",
        swoff_ch="32 swoff",
        joint_ch="17 ankle ang)",
        joint_name="Ankle",
        condition=condition,
    )

    # # For Mouse 2
    # axd["hip_two"] = joint_fig(
    #     kindf,
    #     swon_ch="31 swon",
    #     swoff_ch="32 swoff",
    #     joint_ch="15 hip ang",
    #     joint_name="Hip",
    #     condition=condition,
    # )
    #
    # # For Knee
    # axd["knee_two"] = joint_fig(
    #     kindf,
    #     swon_ch="31 swon",
    #     swoff_ch="32 swoff",
    #     joint_ch="16 knee ang",
    #     joint_name="Knee",
    #     condition=condition,
    # )
    #
    # # For Ankle
    # axd["ankle_two"] = joint_fig(
    #     kindf,
    #     swon_ch="31 swon",
    #     swoff_ch="32 swoff",
    #     joint_ch="17 ankle ang)",
    #     joint_name="Ankle",
    #     condition=condition,
    # )

    # fig.set_size()

    fig = plt.gcf()
    fig.set_size_inches(8, 10)
    fig.tight_layout()

    if save_plots is True:
        plt.savefig(figure_filename, dpi=300)

    plt.show()


if __name__ == "__main__":
    main()
