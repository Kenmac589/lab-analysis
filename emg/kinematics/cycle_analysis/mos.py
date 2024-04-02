import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns

import latstability as ls


def double_support_timings(input_dataframe, ds_channel):
    """Timings of periods of double support for one side
    @param input_dataframe: spike file input as *.csv
    @param ds_channel: the channel with double support markers

    @return ds_timings: List of timings where double support occurs
    """

    # Define the value and column to search for
    value_to_find = 0
    column_to_search = ds_channel
    # stance_end = swonset_channel
    # column_to_search = swonset_channel
    column_for_time = "Time"
    # column_for_treadmill = "2 Trdml"

    # Store time values and treadmill speed when the specified value is found
    time_values = []

    # Iterate through the DataFrame and process matches
    for index, row in input_dataframe.iterrows():
        if row[column_to_search] != value_to_find:
            time_value = row[column_for_time]
            time_values.append(time_value)

    ds_timings = time_values

    return ds_timings


def cop_filter(input_dataframe, ds_timings, cop_channel):
    """Stance duration during step cycle
    @param input_dataframe: spike file input as *.csv
    @param ds_channel: the channel with double support markers

    @return ds_timings: List of timings where double support occurs
    """

    # Filtering whole dataframe down to values we are considering
    input_dataframe_subset = input_dataframe.loc[:, ["Time", cop_channel]]
    input_dataframe_subset = input_dataframe_subset.set_index("Time")

    # Getting real mos values to apply over total array
    mos_values_original = input_dataframe_subset[cop_channel].values
    mos_values = input_dataframe_subset.loc[ds_timings, :][cop_channel].values

    # Get indices which are not actually apart of DS phases
    indices = np.where(~np.in1d(mos_values_original, mos_values))[0]

    # Replace non-matching values in original with NaN
    mos_values_original[indices] = np.nan

    # Applying real values as a mask to maintain the same length of the array and replace with nans

    return mos_values_original


def manual_marks(related_trace, title="Select Points"):
    """Manually annotate points of interest on a given trace
    :param related_trace: Trace you want to annotate

    :return manual_marks_x: array of indices to approx desired value in original trace
    :return manual_marks_y: array of selected values
    """

    # Open interface with trace
    plt.plot(related_trace)
    plt.title(title)

    # Go through and label regions desired
    manual_marks_pair = plt.ginput(0, 0)

    # Store x coordinates as rounded off ints to be used as indices
    manual_marks_x = np.asarray(list(map(lambda x: x[0], manual_marks_pair)))
    manual_marks_x = manual_marks_x.astype(np.int32)

    # Store y coordinates as the actual value desired
    manual_marks_y = np.asarray(list(map(lambda x: x[1], manual_marks_pair)))
    plt.show()

    return manual_marks_x, manual_marks_y


def main():

    # wt1nondf = pd.read_csv("./wt-1_non-perturbation-cop.txt", delimiter=",", header=0)
    # wt1perdf = pd.read_csv("./wt-1-perturbation-xcom.txt", delimiter=",", header=0)
    # wt3nondf = pd.read_csv("./wt-3-non-perturbation-xcom.txt", delimiter=",", header=0)
    # wt3perdf = pd.read_csv("./wt-3-perturbation-xcom.txt", delimiter=",", header=0)
    # wt4nondf = pd.read_csv("./wt-4-non-perturbation-xcom.txt", delimiter=",", header=0)
    wt4perdf = pd.read_csv("./wt-4-perturbation-xcom.txt", delimiter=",", header=0)

    # Grabbing individual channels
    # wt1per_xcom = wt1perdf["67 xCoM"].to_numpy(dtype=float)
    # wt1per_leftcop = wt1perdf["v5 L COP"].to_numpy(dtype=float)
    # wt1per_rightcop = wt1perdf["v3 R COP"].to_numpy(dtype=float)

    wt4per_xcom = wt4perdf["v1 xCoM"].to_numpy(dtype=float)
    wt4per_leftcop = wt4perdf["v3 L COP"].to_numpy(dtype=float)
    wt4per_rightcop = wt4perdf["v2 R COP"].to_numpy(dtype=float)

    # Grabbing peaks again simply for demonstration
    wt4per_xcom_peaks, _ = sp.signal.find_peaks(wt4per_xcom, width=40)
    wt4per_xcom_troughs, _ = sp.signal.find_peaks(-wt4per_xcom, width=40)

    # Manual example
    # wt4per_xcom_peaks, _ = manual_marks(wt4per_xcom, title="Select Peaks")
    # wt4per_xcom_troughs, _ = manual_marks(wt4per_xcom, title="Select Troughs")

    # wt1per_xcom_peaks, _ = sp.signal.find_peaks(wt1per_xcom, width=40)
    # wt1per_xcom_troughs, _ = sp.signal.find_peaks(wt1per_xcom, width=40)

    # Calculating margin of stability
    wt4per_lmos, wt4per_rmos = ls.mos(
        wt4per_xcom, wt4per_leftcop, wt4per_rightcop, manual_peaks=False
    )

    # wt1per_lmos, wt1per_rmos = mos(
    #     wt1per_xcom, wt1per_leftcop, wt1per_rightcop, manual_peaks=True
    # )

    # man_mark_xcords, man_mark_val = manual_marks(wt4per_xcom)
    # print(f"These are some example coordinates when done manually {man_mark_xcords}")
    # print()
    # print(f"Here are the automatically computed one {wt4per_xcom_peaks}")

    # Plotting
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set(style="white", font_scale=1.0, rc=custom_params)

    # Figure for M4 perturbation
    xcom_legend = [
        "xCoM",
        "xCoM peaks",
        "xCoM troughs",
        "L COP",
        "R COP",
    ]
    mos_legend = ["L MoS", "R MoS"]
    fig, axs = plt.subplots(2)
    fig.suptitle("Measurement of Stability WT M4 with Perturbation")

    # For plotting figure demonstrating how calculation was done
    axs[0].set_title("How MoS is Derived")
    axs[0].plot(wt4per_xcom)
    axs[0].plot(wt4per_xcom_peaks, wt4per_xcom[wt4per_xcom_peaks], "^")
    axs[0].plot(wt4per_xcom_troughs, wt4per_xcom[wt4per_xcom_troughs], "v")
    axs[0].plot(
        wt4per_leftcop,
    )
    axs[0].plot(wt4per_rightcop)
    axs[0].legend(xcom_legend, bbox_to_anchor=(1, 0.7))

    # Looking at result
    axs[1].set_title("MoS Result")
    axs[1].bar(0, np.mean(wt4per_lmos), yerr=np.std(wt4per_lmos), capsize=5)
    axs[1].bar(1, np.mean(wt4per_rmos), yerr=np.std(wt4per_rmos), capsize=5)
    axs[1].legend(mos_legend, bbox_to_anchor=(1, 0.7))

    plt.tight_layout()
    plt.show()

    # Figure for M1 perturbation
    # xcom_legend = [
    #     "xCoM",
    #     "xCoM peaks",
    #     "xCoM troughs",
    #     "L COP",
    #     "R COP",
    # ]
    # mos_legend = ["L MoS", "R MoS"]
    # fig, axs = plt.subplots(2)
    # fig.suptitle("Measurement of Stability WT M1 with Perturbation")
    #
    # # For plotting figure demonstrating how calculation was done
    # axs[0].set_title("How MoS is Derived")
    # axs[0].plot(wt4per_xcom)
    # axs[0].plot(wt4per_xcom_peaks, wt4per_xcom[wt4per_xcom_peaks], "^")
    # axs[0].plot(wt4per_xcom_troughs, wt4per_xcom[wt4per_xcom_troughs], "v")
    # axs[0].plot(wt4per_leftcop)
    # axs[0].plot(wt4per_rightcop)
    # axs[0].legend(xcom_legend, bbox_to_anchor=(1, 0.7))
    #
    # # Looking at result
    # axs[1].set_title("MoS Result")
    # axs[1].bar(0, np.mean(wt4per_lmos), yerr=np.std(wt4per_lmos), capsize=5)
    # axs[1].bar(1, np.mean(wt4per_rmos), yerr=np.std(wt4per_rmos), capsize=5)
    # axs[1].legend(mos_legend, bbox_to_anchor=(1, 0.7))
    #
    # plt.tight_layout()
    # plt.show()

    #
    # # Saving results
    # # NOTE: Make sure that you are naming the correct variables to be saved
    #
    # np.savetxt("./wt4per_lmos.csv", wt4per_lmos, delimiter=",")
    # np.savetxt("./wt4per_rmos.csv", wt4per_lmos, delimiter=",")
    #


if __name__ == "__main__":
    main()
