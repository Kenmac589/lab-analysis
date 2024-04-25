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

    # Need to get hip heights for xcom
    # dtrpre_2non = pd.read_csv(
    #     "./dtr_data/predtx/dtr-pre-2-non-all.txt", delimiter=",", header=0
    # )
    # dtrpre_2per = pd.read_csv(
    #     "./dtr_data/predtx/dtr-pre-2-per-all.txt", delimiter=",", header=0
    # )
    # dtrpre_2sin = pd.read_csv(
    #     "./dtr_data/predtx/dtr-pre-2-sin-all.txt", delimiter=",", header=0
    # )
    #
    # dtrpre_3non = pd.read_csv(
    #     "./dtr_data/predtx/dtr-pre-3-non-all.txt", delimiter=",", header=0
    # )
    # dtrpre_3per = pd.read_csv(
    #     "./dtr_data/predtx/dtr-pre-3-per-all.txt", delimiter=",", header=0
    # )
    # dtrpre_3sin = pd.read_csv(
    #     "./dtr_data/predtx/dtr-pre-3-sin-all-2.txt", delimiter=",", header=0
    # )
    # dtrpre_3non = pd.read_csv(
    #     "./dtr_data/predtx/dtr-pre-3-non-all.txt", delimiter=",", header=0
    # )
    # dtrpre_5non = pd.read_csv(
    #     "./dtr_data/predtx/dtr-pre-5-non-all.txt", delimiter=",", header=0
    # )
    # dtrpre_5per = pd.read_csv(
    #     "./dtr_data/predtx/dtr-pre-5-per-all-2.txt", delimiter=",", header=0
    # )
    dtrpre_5sin = pd.read_csv(
        "./dtr_data/predtx/dtr-pre-5-sin-all.txt", delimiter=",", header=0
    )

    # For MoS

    # Some things to set for plotting/saving
    manual_analysis = False
    save_auto = False
    lmos_filename = "./dtr_data/predtx/predtx_5sin_lmos.csv"
    rmos_filename = "./dtr_data/predtx/predtx_5sin_rmos.csv"
    figure_title = (
        "Measurement of Stability For DTR M5 with Sinusoidal Perturbation pre-DTX"
    )

    # Grabbing individual channels
    xcom = dtrpre_5sin["v1 xCoM"].to_numpy(dtype=float)
    leftcop = dtrpre_5sin["v3 L COP"].to_numpy(dtype=float)
    rightcop = dtrpre_5sin["v2 R COP"].to_numpy(dtype=float)
    left_DS = dtrpre_5sin["50 LDS cle"].to_numpy(dtype=float)
    right_DS = dtrpre_5sin["49 RDS cle"].to_numpy(dtype=float)

    # Remove periods where it is not present or not valid
    # leftcop = np.where(leftcop == 0.0, np.nan, leftcop)
    rightcop = np.where(rightcop == 0.0, np.nan, rightcop)
    left_band = np.percentile(xcom, q=50)
    # right_band = 2
    # rightcop[rightcop < right_band] = np.nan
    leftcop[leftcop < left_band] = np.nan

    lmos, rmos, xcom_peaks, xcom_troughs = ls.mos(
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
    # plt.savefig("./dtr-mos-output.pdf", dpi=300)
    plt.show()

    # Saving results
    if manual_analysis is True:
        np.savetxt(lmos_filename, lmos, delimiter=",")
        np.savetxt(rmos_filename, rmos, delimiter=",")
        print("mos results saved!")
    elif manual_analysis is False and save_auto is True:
        np.savetxt(lmos_filename, lmos, delimiter=",")
        np.savetxt(rmos_filename, rmos, delimiter=",")
        print("mos results saved!")
    else:
        print("mos results not saved")


if __name__ == "__main__":
    main()
