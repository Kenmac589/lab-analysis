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


def peak_grabber_test(related_trace, width_threshold=35):

    # Getting peaks and troughs
    xcom = related_trace
    xcom_peaks, _ = sp.signal.find_peaks(xcom, width=width_threshold)
    xcom_troughs, _ = sp.signal.find_peaks(-xcom, width=width_threshold)

    plt.plot(xcom)
    plt.plot(xcom_peaks, xcom[xcom_peaks], "^")
    plt.plot(xcom_troughs, xcom[xcom_troughs], "v")
    plt.show()


def mos_manual_marks(related_trace, leftcop, rightcop, title="Select Points"):
    """Manually annotate points of interest on a given trace
    :param related_trace: Trace you want to annotate

    :return manual_marks_x: array of indices to approx desired value in original trace
    :return manual_marks_y: array of selected values
    """

    # Removing 0 values
    rightcop = np.where(rightcop == 0.0, np.nan, rightcop)
    leftcop = np.where(leftcop == 0.0, np.nan, leftcop)

    # Correcting to DS regions are close to label
    left_adjustment = np.mean(related_trace) + 0.5
    right_adjustment = np.mean(related_trace) - 0.5

    rightcop = rightcop * right_adjustment
    leftcop = leftcop * left_adjustment

    # Open interface with trace
    plt.plot(related_trace)
    plt.plot(leftcop)
    plt.plot(rightcop)
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


def mos(
    xcom, leftcop, rightcop, leftds, rightds, manual_peaks=False, width_threshold=40
):

    # Remove periods where it is not present or not valid
    left_band = np.percentile(xcom, q=50)
    rightcop = np.where(rightcop == 0.0, np.nan, rightcop)
    # rightcop[rightcop < right_band] = np.nan
    leftcop[leftcop < left_band] = np.nan

    # Optional manual point selection
    if manual_peaks is False:
        # Getting peaks and troughs
        xcom_peaks, _ = sp.signal.find_peaks(xcom, width=width_threshold)
        xcom_troughs, _ = sp.signal.find_peaks(-xcom, width=width_threshold)
    elif manual_peaks is True:
        xcom_peaks, _ = mos_manual_marks(xcom, leftds, rightds, title="Select Peaks")
        xcom_troughs, _ = mos_manual_marks(
            xcom, leftds, rightds, title="Select Troughs"
        )
    else:
        print("The `manual` variable must be a boolean")

    lmos_values = np.array([])
    rmos_values = np.array([])

    for i in range(len(xcom_peaks) - 1):
        # Getting window between peak values
        beginning = xcom_peaks[i]
        end = xcom_peaks[i + 1]
        region_to_consider = leftcop[beginning:end]

        # Getting non-nan values from region
        value_cop = region_to_consider[~np.isnan(region_to_consider)]

        # Making sure we are actually grabbing the last meaningful region of center of pressure
        if value_cop.shape[0] >= 2:
            cop_point = np.mean(value_cop)
            lmos = cop_point - xcom[beginning]
            lmos_values = np.append(lmos_values, lmos)

    for i in range(len(xcom_troughs) - 1):
        # Getting window between peak values
        beginning = xcom_troughs[i]
        end = xcom_troughs[i + 1]
        region_to_consider = rightcop[beginning:end]

        # Getting non-nan values from region
        value_cop = region_to_consider[~np.isnan(region_to_consider)]
        if value_cop.shape[0] >= 2:
            cop_point = np.mean(value_cop)
            rmos = xcom[beginning] - cop_point
            rmos_values = np.append(rmos_values, rmos)

    return lmos_values, rmos_values, xcom_peaks, xcom_troughs


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


def main():

    # For Egr3 KO mice
    # egr3_6nondf = pd.read_csv("./egr3-6-non-xcom-redo.txt", delimiter=",", header=0)
    # egr3_6perdf = pd.read_csv("./egr3-6-per-xcom-redo.txt", delimiter=",", header=0)
    # egr3_6sindf = pd.read_csv("./egr3-6-sinus-xcom-redo.txt", delimiter=",", header=0)
    # egr3_7nondf = pd.read_csv("./egr3-7-non-xcom-redo.txt", delimiter=",", header=0)
    # egr3_7perdf = pd.read_csv("./egr3-7-per-xcom-redo.txt", delimiter=",", header=0)
    # egr3_7sindf = pd.read_csv("./egr3-7-sinus-xcom-redo.txt", delimiter=",", header=0)
    # egr3_8nondf = pd.read_csv("./egr3-8-non-xcom-redo.txt", delimiter=",", header=0)
    # egr3_8perdf = pd.read_csv("./egr3-8-per-xcom-redo.txt", delimiter=",", header=0)
    # egr3_8sindf = pd.read_csv("./egr3-8-sinus-xcom-redo.txt", delimiter=",", header=0)
    # egr3_9nondf = pd.read_csv("./egr3-9-non-xcom-redo.txt", delimiter=",", header=0)
    # egr3_9perdf = pd.read_csv("./egr3-9-per-xcom-redo-2.txt", delimiter=",", header=0)
    # egr3_9sindf = pd.read_csv("./egr3-9-sinus-xcom-redo-2.txt", delimiter=",", header=0)
    # egr3_10nondf = pd.read_csv("./egr3-10-non-xcom-redo.txt", delimiter=",", header=0)
    egr3_10perdf = pd.read_csv(
        "./egr3-10-per-xcom-redo-pt2.txt", delimiter=",", header=0
    )
    # egr3_10sindf = pd.read_csv("./egr3-10-sinus-xcom.txt", delimiter=",", header=0)

    # Some things to set for plotting/saving
    manual_analysis = True
    save_auto = False
    lmos_filename = "./egr3_10per_lmos-2.csv"
    rmos_filename = "./egr3_10per_rmos-2.csv"
    figure_title = "Measurement of Stability For Egr3 KO M10 with Perturbation (2)"

    # Grabbing individual channels
    xcom = egr3_10perdf["v1 xCoM"].to_numpy(dtype=float)
    leftcop = egr3_10perdf["v3 L COP"].to_numpy(dtype=float)
    rightcop = egr3_10perdf["v2 R COP"].to_numpy(dtype=float)
    left_DS = egr3_10perdf["67 LDS cle"].to_numpy(dtype=float)
    right_DS = egr3_10perdf["66 RDS cle"].to_numpy(dtype=float)

    # Remove periods where it is not present or not valid
    left_band = np.percentile(xcom, q=50)
    right_band = 2
    rightcop = np.where(rightcop == 0.0, np.nan, rightcop)
    rightcop[rightcop < right_band] = np.nan
    leftcop[leftcop < left_band] = np.nan

    lmos, rmos, xcom_peaks, xcom_troughs = mos(
        xcom,
        leftcop,
        rightcop,
        left_DS,
        right_DS,
        manual_peaks=manual_analysis,
    )

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
    fig.suptitle(figure_title)

    # For plotting figure demonstrating how calculation was done
    axs[0].set_title("How MoS is Derived")
    axs[0].plot(xcom)
    axs[0].plot(xcom_peaks, xcom[xcom_peaks], "^")
    axs[0].plot(xcom_troughs, xcom[xcom_troughs], "v")
    axs[0].plot(leftcop)
    axs[0].plot(rightcop)
    axs[0].legend(xcom_legend, bbox_to_anchor=(1, 0.7))

    # Looking at result
    axs[1].set_title("MoS Result")
    axs[1].bar(0, np.mean(lmos), yerr=np.std(lmos), capsize=5)
    axs[1].bar(1, np.mean(rmos), yerr=np.std(rmos), capsize=5)
    axs[1].legend(mos_legend, bbox_to_anchor=(1, 0.7))

    plt.tight_layout()
    plt.show()

    # Saving results
    if manual_analysis is True:
        np.savetxt(lmos_filename, lmos, delimiter=",")
        np.savetxt(rmos_filename, rmos, delimiter=",")
    elif manual_analysis is False and save_auto is True:
        np.savetxt(lmos_filename, lmos, delimiter=",")
        np.savetxt(rmos_filename, rmos, delimiter=",")
    else:
        print("not saved")


if __name__ == "__main__":
    main()
