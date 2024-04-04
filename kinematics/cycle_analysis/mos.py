import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import scipy as sp
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


def main():

    # For WT Group

    # wt1nondf = pd.read_csv("./wt-1_non-perturbation-cop.txt", delimiter=",", header=0)
    # wt1perdf = pd.read_csv("./wt-1-perturbation-xcom.txt", delimiter=",", header=0)
    # wt1sindf = pd.read_csv("./wt-1-sinus.txt", delimiter=",", header=0)
    # wt2nondf = pd.read_csv("./wt-2-non-perturbation-xcom.txt", delimiter=",", header=0)
    # wt2perdf = pd.read_csv("./wt-2-perturbation-xcom.txt", delimiter=",", header=0)
    # wt2sindf = pd.read_csv("./wt-2-sinus.txt", delimiter=",", header=0)
    # wt3nondf = pd.read_csv("./wt-3-non-perturbation-xcom.txt", delimiter=",", header=0)
    # wt3perdf = pd.read_csv("./wt-3-perturbation-xcom.txt", delimiter=",", header=0)
    # wt3sindf = pd.read_csv("./wt-3-sinus.txt", delimiter=",", header=0)
    # wt4nondf = pd.read_csv("./wt-4-non-perturbation-xcom.txt", delimiter=",", header=0)
    # wt4perdf = pd.read_csv("./wt-4-perturbation-xcom.txt", delimiter=",", header=0)
    # wt4sindf = pd.read_csv("./wt-4-sinus.txt", delimiter=",", header=0)
    # wt5nondf = pd.read_csv("./wt-5-non-perturbation-xcom.txt", delimiter=",", header=0)
    # wt5perdf = pd.read_csv("./wt-5-non-perturbation-xcom.txt", delimiter=",", header=0)

    # For Egr3 KO mice
    egr3_6nondf = pd.read_csv(
        "./egr3-6-non-perturbation-xcom.txt", delimiter=",", header=0
    )

    # Grabbing individual channels
    egr3_6non_xcom = egr3_6nondf["v1 xCoM"].to_numpy(dtype=float)
    egr3_6non_leftcop = egr3_6nondf["v3 L COP"].to_numpy(dtype=float)
    egr3_6non_rightcop = egr3_6nondf["v2 R COP"].to_numpy(dtype=float)

    # wt2per_xcom = wt2perdf["v1 xCoM"].to_numpy(dtype=float)
    # wt2per_leftcop = wt2perdf["v3 L COP"].to_numpy(dtype=float)
    # wt2per_rightcop = wt2perdf["v2 R COP"].to_numpy(dtype=float)

    # Calculating margin of stability
    egr3_6non_lmos, egr3_6non_rmos, egr3_6non_xcom_peaks, egr3_6non_xcom_troughs = (
        ls.mos(
            egr3_6non_xcom,
            egr3_6non_leftcop,
            egr3_6non_rightcop,
            manual_peaks=True,
        )
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
    fig.suptitle("Measurement of Stability For Egr3 KO without Perturbation")

    # For plotting figure demonstrating how calculation was done
    axs[0].set_title("How MoS is Derived")
    axs[0].plot(egr3_6non_xcom)
    axs[0].plot(egr3_6non_xcom_peaks, egr3_6non_xcom[egr3_6non_xcom_peaks], "^")
    axs[0].plot(egr3_6non_xcom_troughs, egr3_6non_xcom[egr3_6non_xcom_troughs], "v")
    axs[0].plot(
        egr3_6non_leftcop,
    )
    axs[0].plot(egr3_6non_rightcop)
    axs[0].legend(xcom_legend, bbox_to_anchor=(1, 0.7))

    # Looking at result
    axs[1].set_title("MoS Result")
    axs[1].bar(0, np.mean(egr3_6non_lmos), yerr=np.std(egr3_6non_lmos), capsize=5)
    axs[1].bar(1, np.mean(egr3_6non_rmos), yerr=np.std(egr3_6non_rmos), capsize=5)
    axs[1].legend(mos_legend, bbox_to_anchor=(1, 0.7))

    plt.tight_layout()
    # plt.show()

    # Saving results
    np.savetxt("./egr3_6non_lmos.csv", egr3_6non_lmos, delimiter=",")
    np.savetxt("./egr3_6non_rmos.csv", egr3_6non_rmos, delimiter=",")
    #


if __name__ == "__main__":
    main()
