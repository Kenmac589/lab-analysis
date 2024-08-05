import csv

import dlc2kinematics as dlck
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from kinsynpy import dlctools as dlt

# from dlc2kinematics import Visualizer2D


def main():

    # NOTE: Very important this is checked before running

    mouse_number = 2
    video = "20"
    condition = "sin"
    hiph_entry = f"12mo-predtx-{mouse_number}-{condition}-{video}"
    manual_analysis = False
    save_auto = False
    filter_k = 14

    print(f"Analysis for {hiph_entry}")

    # Loading main kinematic dataset
    df, bodyparts, scorer = dlck.load_data(
        f"./aging/12mo/1yr-dtr_norosa-2/1yr-dtr_norosa-2-predtx/1yrDTRnoRosa-M2-preDTX-23102023_0000{video}DLC_resnet50_1yrDTRnoRosa-preDTXJan31shuffle1_1030000_filtered.h5"
    )

    # Loading in skeleton
    # sk_df = pd.read_csv(f"./aging/12mo/1yr-dtr_norosa-1/1yr-dtr_norosa-1-predtx/1yrDTRnoRosa-M1-19102023_000001DLC_resnet50_1yrDTRnoRosa-preDTXJan31shuffle1_1030000_filtered_labeled.mp4")

    # Settings before running initial workup from DeepLabCut
    figure_title = (
        f"Step Cycles for 1yr DTR-noRosa pre-DTX M{mouse_number}-{condition}-{video}"
    )
    figure_filename = f"./aging/12mo/aging-12mo-figures/12mo-dtr_norosa-predtx-m{mouse_number}-{condition}-{video}.svg"
    step_cycles_filename = f"./aging/12mo/aging-12mo-saved_values/12mo-dtr_norosa-predtx-m{mouse_number}-{condition}-step-cycles-{video}.csv"

    # Some things to set for plotting/saving
    lmos_filename = f"./aging/12mo/aging-12mo-saved_values/12mo-dtr_norosa-predtx-m{mouse_number}-{condition}-lmos-{video}.csv"
    rmos_filename = f"./aging/12mo/aging-12mo-saved_values/12mo-dtr_norosa-predtx-m{mouse_number}-{condition}-rmos-{video}.csv"

    mos_figure_title = (
        f"Measurement of Stability For 12 month old DTR pre-DTX M{mouse_number}-{video}"
    )
    mos_figure_filename = f"./aging/12mo/aging-12mo-figures/12mo-dtr_norosa-predtx-m{mouse_number}-{video}-mos_analysis.svg"
    calib_markers = [
        "calib_1",
        "calib_2",
        "calib_3",
        "calib_4",
        "calib_5",
        "calib_6",
    ]

    # For visualizing skeleton
    # config_path = (
    #     "../../deeplabcut/1yr/1yrDTRnoRosa-preDTX-kenzie-2024-01-31_analyzed/config.yaml"
    # )
    # foi = f"./aging/12mo/1yr-dtr_norosa-1/1yr-dtr_norosa-1-predtx/non_filtered/1yrDTRnoRosa-M1-19102023_00000{video}DLC_resnet50_1yrDTRnoRosa-preDTXJan31shuffle1_1030000.h5"
    # viz = Visualizer2D(config_path, foi, form_skeleton=True)
    # viz.view(show_axes=True, show_grid=True, show_labels=True)
    # plt.show()

    calib_factor = dlt.dlc_calibrate(df, bodyparts, scorer, calib_markers)

    # Grabbing toe marker data
    toe = df[scorer]["toe"]
    hip = df[scorer]["hip"]
    lhl = df[scorer]["mirror_lhl"]
    rhl = df[scorer]["mirror_rhl"]
    lfl = df[scorer]["mirror_lfl"]
    rfl = df[scorer]["mirror_rfl"]
    com = df[scorer]["mirror_com"]

    # Converting to numpy array
    toe_np = pd.array(toe["x"])
    toe_np = toe_np / calib_factor
    toey_np = pd.array(toe["y"])
    toey_np = toey_np / calib_factor
    hipy_np = pd.array(hip["y"])
    hipy_np = hipy_np / calib_factor

    comy_np = pd.array(com["y"])
    comy_np = comy_np / calib_factor
    time = np.arange(0, len(comy_np), 1)
    # time_vec = np.vectorize(frame_to_time)
    # time = time_vec(time)
    time = dlt.frame_to_time(time)
    rfl_np = pd.array(rfl["y"])
    rfl_np = rfl_np / calib_factor
    rhl_np = pd.array(rhl["y"])
    rhl_np = rhl_np / calib_factor
    lfl_np = pd.array(lfl["y"])
    lfl_np = lfl_np / calib_factor
    lhl_np = pd.array(lhl["y"])
    lhl_np = lhl_np / calib_factor

    # Filtering to clean up traces like you would in spike
    toe_smooth = dlt.median_filter(toe_np, filter_k)
    toe_smooth = sp.signal.savgol_filter(toe_smooth, 20, 3)
    # com_med = dlt.median_filter(comy_np, filter_k)
    com_med = sp.signal.savgol_filter(comy_np, 40, 3)

    rfl_med = dlt.median_filter(rfl_np, filter_k)
    rfl_med = sp.signal.savgol_filter(rfl_med, 30, 3)
    rhl_med = dlt.median_filter(rhl_np, filter_k)
    rhl_med = sp.signal.savgol_filter(rhl_med, 30, 3)
    lfl_med = dlt.median_filter(lfl_np, filter_k)
    lfl_med = sp.signal.savgol_filter(lfl_med, 30, 3)
    lhl_med = dlt.median_filter(lhl_np, filter_k)
    lhl_med = sp.signal.savgol_filter(lhl_med, 30, 3)

    # Cleaning up selection to region before mouse moves back
    # toe_roi_selection_fil = toe_filtered[0:2550]

    # rfl_med = rfl_med[1400:]
    # rhl_med = rhl_med[1400:]
    # lfl_med = lfl_med[1400:]
    # lhl_med = lhl_med[1400:]
    time_trimmed = time
    # toe_smooth = toe_smooth[1400:]
    com_trimmed = com_med

    # Center of pressures
    com_slope = dlt.spike_slope(com_trimmed, 30)
    hip_h = dlt.hip_height(toey_np, hipy_np)
    print(f"Calculated hip height: {hip_h}")
    xcom_trimmed = dlt.xcom(com_trimmed, com_slope, hip_h)

    # Experimental Estimation of CoP considering the standards used
    rightcop = dlt.cop(rfl_med, rhl_med)
    rightcop = sp.signal.savgol_filter(rightcop, 40, 3)
    leftcop = dlt.cop(lfl_med, lhl_med)
    leftcop = sp.signal.savgol_filter(leftcop, 40, 3)
    right_DS = rightcop
    left_DS = leftcop

    # Calling function for swing estimation
    swing_onset, swing_offset = dlt.swing_estimation(toe_smooth)
    step_cyc_durations = dlt.step_cycle_est(toe_smooth)

    # Calling function for step cycle calculation

    # Some of my default plotting parameters I like
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set(
        style="white",
        font_scale=1.6,
        font="serif",
        palette="colorblind",
        rc=custom_params,
    )

    # Plot Legend
    swing_legend = [
        "Limb X cord",
        "Swing offset",
        "Swing onset",
    ]
    filtest_legend = [
        # "Original",
        # "Median",
        "xCoM",
        "CoMy",
        "L CoP",
        "R CoP",
        # "Slope",
    ]

    fig, axs = plt.subplots(2)
    fig.suptitle(figure_title)

    # Showing results for step cycle timing
    axs[0].set_title("Filter test")
    # axs[0].plot(comy_np)
    # axs[0].plot(com_med)
    axs[0].plot(time_trimmed, xcom_trimmed)
    axs[0].plot(time_trimmed, com_trimmed)
    axs[0].plot(time_trimmed, leftcop)
    axs[0].plot(time_trimmed, rightcop)
    # axs[0].plot(time_trimmed, com_slope)
    axs[0].legend(filtest_legend, loc="best")
    # axs[0].bar(0, np.mean(step_cyc_durations), yerr=np.std(step_cyc_durations), capsize=5)

    # For plotting figure demonstrating how swing estimation was done
    axs[1].set_title("Swing Estimation")
    axs[1].plot(toe_smooth)
    axs[1].plot(swing_offset, toe_smooth[swing_offset], "^")
    axs[1].plot(swing_onset, toe_smooth[swing_onset], "v")
    axs[1].legend(swing_legend, loc="best")

    # Saving Figure in same folder
    fig = mpl.pyplot.gcf()
    fig.set_size_inches(19.8, 10.80)

    if manual_analysis is True:
        # Saving plot and results
        np.savetxt(step_cycles_filename, step_cyc_durations, delimiter=",")
        plt.savefig(figure_filename, dpi=300)
        print("Kinematic results saved")
    elif manual_analysis is False and save_auto is True:
        np.savetxt(step_cycles_filename, step_cyc_durations, delimiter=",")
        plt.savefig(figure_filename, dpi=300)
        print("Kinematic results saved")
    else:
        print("Kinematic results not saved")
        plt.show()

    # Now onto Lateral stability

    lmos, rmos, xcom_peaks, xcom_troughs = dlt.mos(
        xcom_trimmed,
        leftcop,
        rightcop,
        left_DS,
        right_DS,
        manual_peaks=manual_analysis,
        width_threshold=50,
    )
    lmos = np.where(lmos < 0.0, np.nan, lmos)
    rmos = np.where(rmos < 0.0, np.nan, rmos)

    mos_comb = pd.DataFrame(columns=["Limb", "MoS (cm)"])
    for i in range(len(lmos)):
        condition = "Left"
        fixed_array = lmos.ravel()
        mos_entry = [[condition, fixed_array[i]]]
        mos_comb = mos_comb._append(
            pd.DataFrame(mos_entry, columns=["Limb", "MoS (cm)"]),
            ignore_index=True,
        )

    for i in range(len(rmos)):
        condition = "Right"
        fixed_array = rmos.ravel()
        mos_entry = [[condition, fixed_array[i]]]
        mos_comb = mos_comb._append(
            pd.DataFrame(mos_entry, columns=["Limb", "MoS (cm)"]),
            ignore_index=True,
        )

    xcom_legend = [
        "xCoM",
        "xCoM peaks",
        "xCoM troughs",
        "L COP",
        "R COP",
    ]
    fig = plt.figure(figsize=(15.8, 10.80))
    axs = fig.subplot_mosaic([["mos_calc", "mos_calc"], ["mos_violin", "mos_box"]])

    fig.suptitle(mos_figure_title)
    # For plotting figure demonstrating how calculation was done
    axs["mos_calc"].set_title("How MoS is Derived")
    axs["mos_calc"].plot(xcom_trimmed)
    axs["mos_calc"].plot(xcom_peaks, xcom_trimmed[xcom_peaks], "^")
    axs["mos_calc"].plot(xcom_troughs, xcom_trimmed[xcom_troughs], "v")
    axs["mos_calc"].plot(leftcop)
    axs["mos_calc"].plot(rightcop)
    axs["mos_calc"].legend(xcom_legend, bbox_to_anchor=(1, 0.7))

    # Looking at result
    # axs[1].set_title("MoS Result")
    sns.barplot(data=mos_comb, x="Limb", y="MoS (cm)", capsize=0.04, ax=axs["mos_box"])
    sns.violinplot(
        data=mos_comb, x="Limb", y="MoS (cm)", inner="point", ax=axs["mos_violin"]
    )
    # axs[1].bar(0, np.mean(lmos), yerr=np.std(lmos), capsize=5)
    # axs[1].bar(1, np.mean(rmos), yerr=np.std(rmos), capsize=5)
    # axs[1].legend(bbox_to_anchor=(1, 0.7))

    fig = plt.gcf()
    fig.set_size_inches(15.8, 10.80)
    fig.tight_layout()

    # Saving results
    if manual_analysis is True:
        # Saving MoS figure and values
        np.savetxt(lmos_filename, lmos, delimiter=",")
        np.savetxt(rmos_filename, rmos, delimiter=",")
        plt.savefig(mos_figure_filename, dpi=300)

        # Saving hip height to cumulative sheet
        hiph_dict = {hiph_entry: hip_h}
        w = csv.writer(open("./aging/aging-hiph.csv", "a"))
        for key, val in hiph_dict.items():
            w.writerow([key, val])

        print("Mos results saved!")
    elif manual_analysis is False and save_auto is True:
        # Saving MoS figure and values
        np.savetxt(lmos_filename, lmos, delimiter=",")
        np.savetxt(rmos_filename, rmos, delimiter=",")
        plt.savefig(mos_figure_filename, dpi=300)

        # Saving hip height to cumulative sheet
        hiph_dict = {hiph_entry: hip_h}
        w = csv.writer(open("./aging/aging-hiph.csv", "a"))
        for key, val in hiph_dict.items():
            w.writerow([key, val])
        print("Mos results saved!")
    else:
        print("Mos results not saved")
        plt.show()


if __name__ == "__main__":
    main()
