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

    mouse_number = 1
    video = 20
    condition = "sin"
    hiph_entry = f"18mo-predtx-{mouse_number}-{condition}-{video}"
    manual_analysis = False
    save_auto = False
    select_region = False

    # Loading main kinematic dataset
    df, bodyparts, scorer = dlck.load_data(
        f"./aging/18mo/18mo-RAP-predtx/1-6yrRAP-M1-preDTX_0000{video}DLC_resnet50_dtr_update_predtxApr8shuffle1_1110000_filtered.h5"
    )

    # Loading in skeleton
    # sk_df = pd.read_csv(f"./aging/12mo/1yr-dtr_norosa-1/1yr-dtr_norosa-1-predtx/1yrDTRnoRosa-M1-19102023_000001DLC_resnet50_1yrDTRnoRosa-preDTXJan31shuffle1_1030000_filtered_labeled.mp4")

    # Settings before running initial workup from DeepLabCut
    figure_title = (
        f"Step Cycles for 18mo RAP pre-DTX M{mouse_number}-{condition}-{video}"
    )
    figure_filename = f"./aging/18mo/aging-18mo-figures/18mo-rap-predtx-m{mouse_number}-{condition}-{video}.svg"
    step_cycles_filename = f"./aging/18mo/aging-18mo-saved_values/18mo-rap-predtx-m{mouse_number}-{condition}-step-cycles-{video}.csv"

    # Some things to set for plotting/saving
    lmos_filename = f"./aging/18mo/aging-18mo-saved_values/18mo-rap-predtx-m{mouse_number}-{condition}-lmos-{video}.csv"
    rmos_filename = f"./aging/18mo/aging-18mo-saved_values/18mo-rap-predtx-m{mouse_number}-{condition}-rmos-{video}.csv"

    mos_figure_title = f"Measurement of Stability For 18 month old R.A.P pre-DTX M{mouse_number}-{video}"
    mos_figure_filename = f"./aging/18mo/aging-18mo-figures/18mo-rap-predtx-m{mouse_number}-{video}-mos_analysis.svg"
    calib_markers = [
        "calib_1",
        "calib_2",
        "calib_3",
        "calib_4",
        "calib_5",
        "calib_6",
    ]

    calib_factor = dlt.dlc_calibrate(df, bodyparts, scorer, calib_markers)

    # Grabbing marker data
    toex_np = dlt.mark_process(df, scorer, "toe", "x", calib_factor)
    toey_np = dlt.mark_process(df, scorer, "toe", "y", calib_factor)
    hipy_np = dlt.mark_process(df, scorer, "hip", "y", calib_factor)
    comy_np = dlt.mark_process(df, scorer, "mirror_com", "y", calib_factor)
    rfly_np = dlt.mark_process(df, scorer, "mirror_rfl", "y", calib_factor)
    rhly_np = dlt.mark_process(df, scorer, "mirror_rhl", "y", calib_factor)
    lfly_np = dlt.mark_process(df, scorer, "mirror_lfl", "y", calib_factor)
    lhly_np = dlt.mark_process(df, scorer, "mirror_lhl", "y", calib_factor)
    rflx_np = dlt.mark_process(df, scorer, "mirror_rfl", "x", calib_factor)
    rhlx_np = dlt.mark_process(df, scorer, "mirror_rhl", "x", calib_factor)
    lflx_np = dlt.mark_process(df, scorer, "mirror_lfl", "x", calib_factor)
    lhlx_np = dlt.mark_process(df, scorer, "mirror_lhl", "x", calib_factor)
    miry = dlt.mark_process(df, scorer, "mirror", "y", calib_factor, smooth_wind=70)

    # Selecting a Given Region
    if select_region is True:
        reg_start, reg_stop = dlt.analysis_reg_sel(mirror_y=miry, com_y=comy_np)
        toex_np = toex_np[reg_start:reg_stop]
        toey_np = toey_np[reg_start:reg_stop]
        hipy_np = hipy_np[reg_start:reg_stop]
        comy_np = comy_np[reg_start:reg_stop]
        rfly_np = rfly_np[reg_start:reg_stop]
        rhly_np = rhly_np[reg_start:reg_stop]
        lfly_np = lfly_np[reg_start:reg_stop]
        lhly_np = lhly_np[reg_start:reg_stop]
        rflx_np = rflx_np[reg_start:reg_stop]
        rhlx_np = rhlx_np[reg_start:reg_stop]
        lflx_np = lflx_np[reg_start:reg_stop]
        lhlx_np = lhlx_np[reg_start:reg_stop]
    else:
        print("Looking at entire recording")

    # Getting a time adjusted array of equal length for time
    time = np.arange(0, len(comy_np), 1)
    time = dlt.frame_to_time(time)

    # Center of pressures
    com_slope = dlt.spike_slope(comy_np, 30)
    hip_h = dlt.hip_height(toey_np, hipy_np)
    xcom_trimmed = dlt.xcom(comy_np, com_slope, hip_h)

    # Experimental Estimation of CoP considering the standards used
    rightcop = dlt.cop(rfly_np, rhly_np)
    rightcop = sp.signal.savgol_filter(rightcop, 40, 3)
    leftcop = dlt.cop(lfly_np, lhly_np)
    leftcop = sp.signal.savgol_filter(leftcop, 40, 3)
    right_DS = rightcop
    left_DS = leftcop

    # Calling function for swing estimation
    swing_onset, swing_offset = dlt.swing_estimation(toex_np)
    step_cyc_durations = dlt.step_cycle_est(toex_np)

    # Step Width Test
    # fl_stepw = dlt.step_width_est(
    #     rl_x=rflx_np, ll_x=lflx_np, rl_y=rfly_np, ll_y=lfly_np
    # )
    # hl_stepw = dlt.step_width_est(
    #     rl_x=rhlx_np, ll_x=lhlx_np, rl_y=rhly_np, ll_y=lhly_np
    # )

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
    fig, axs = plt.subplots(2)
    fig.suptitle(figure_title)

    # Showing results for step cycle timing
    axs[0].set_title("Filter test")
    # axs[0].plot(comy_np)
    # axs[0].plot(com_med)
    axs[0].plot(time, xcom_trimmed, label="xCoM")
    axs[0].plot(time, comy_np, label="CoM")
    axs[0].plot(time, leftcop, label="L CoP")
    axs[0].plot(time, rightcop, label="R CoP")
    # axs[0].plot(time_trimmed, com_slope)
    axs[0].legend(loc="best")
    # axs[0].bar(0, np.mean(step_cyc_durations), yerr=np.std(step_cyc_durations), capsize=5)

    # For plotting figure demonstrating how swing estimation was done
    axs[1].set_title("Swing Estimation")
    axs[1].plot(toex_np)
    axs[1].plot(swing_offset, toex_np[swing_offset], "^")
    axs[1].plot(swing_onset, toex_np[swing_onset], "v")
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

    fig = plt.figure(figsize=(15.8, 10.80))
    axs = fig.subplot_mosaic([["mos_calc", "mos_calc"], ["mos_violin", "mos_box"]])

    fig.suptitle(mos_figure_title)

    mos_param = {
        "data": mos_comb,
        "x": "Limb",
        "y": "MoS (cm)",
    }

    # For plotting figure demonstrating how calculation was done
    axs["mos_calc"].set_title("How MoS is Derived")
    axs["mos_calc"].plot(xcom_trimmed, label="xCoM")
    axs["mos_calc"].plot(xcom_peaks, xcom_trimmed[xcom_peaks], "^", label="Peaks")
    axs["mos_calc"].plot(xcom_troughs, xcom_trimmed[xcom_troughs], "v", label="Peaks")
    axs["mos_calc"].plot(leftcop, label="L CoP")
    axs["mos_calc"].plot(rightcop, label="R CoP")
    axs["mos_calc"].legend(bbox_to_anchor=(1, 0.7))

    # Looking at results
    sns.boxplot(**mos_param, ax=axs["mos_box"])
    sns.violinplot(**mos_param, inner=None, ax=axs["mos_violin"])
    sns.swarmplot(**mos_param, color="black", ax=axs["mos_box"])
    sns.swarmplot(**mos_param, color="black", ax=axs["mos_violin"])

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
