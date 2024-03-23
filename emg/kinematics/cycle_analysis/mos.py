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


def main():
    wt1nondf = pd.read_csv("./wt-1_non-perturbation-cop.txt", delimiter=",", header=0)
    wt1perdf = pd.read_csv("./wt-1_perturbation-all.txt", delimiter=",", header=0)

    # Grabbing individual channels
    wt1non_xcom = wt1nondf["67 xCoM"].to_numpy(dtype=float)
    wt1non_leftcop = wt1nondf["v5 L COP"].to_numpy(dtype=float)
    wt1non_rightcop = wt1nondf["v3 R COP"].to_numpy(dtype=float)

    wt1per_xcom = wt1perdf["67 xCoM"].to_numpy(dtype=float)
    wt1per_leftcop = wt1perdf["v5 L COP"].to_numpy(dtype=float)
    wt1per_rightcop = wt1perdf["v3 R COP"].to_numpy(dtype=float)
    print(wt1per_rightcop)

    wt1non_xcom_peaks, _ = sp.signal.find_peaks(wt1non_xcom, width=30)
    wt1non_xcom_troughs, _ = sp.signal.find_peaks(-wt1non_xcom, width=30)

    wt1per_xcom_peaks, _ = sp.signal.find_peaks(wt1per_xcom, width=30)
    wt1per_xcom_troughs, _ = sp.signal.find_peaks(-wt1per_xcom, width=30)

    wt1non_lmos, wt1non_rmos = ls.mos(wt1non_xcom, wt1non_leftcop, wt1non_rightcop)
    wt1per_lmos, wt1per_rmos = ls.mos(wt1per_xcom, wt1per_leftcop, wt1per_rightcop)

    # Plotting
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set(style="white", font_scale=1.0, rc=custom_params)

    xcom_legend = [
        "xCoM",
        "xCoM peaks",
        "xCoM troughs",
        "L COP",
        "R COP",
    ]
    mos_legend = ["L MoS", "R MoS"]
    fig, axs = plt.subplots(2)
    fig.suptitle("Measurement of Stability WT M1")

    # For plotting figure demonstrating how calculation was done
    axs[0].set_title("How MoS is Derived")
    axs[0].plot(wt1non_xcom)
    axs[0].plot(wt1non_xcom_peaks, wt1non_xcom[wt1non_xcom_peaks], "^")
    axs[0].plot(wt1non_xcom_troughs, wt1non_xcom[wt1non_xcom_troughs], "v")
    axs[0].plot(wt1non_leftcop)
    axs[0].plot(wt1non_rightcop)
    axs[0].legend(xcom_legend, bbox_to_anchor=(1, 0.7))

    # Looking at result
    axs[1].set_title("MoS Result")
    axs[1].bar(0, np.mean(wt1non_lmos), yerr=np.std(wt1non_lmos), capsize=5)
    axs[1].bar(1, np.mean(wt1non_rmos), yerr=np.std(wt1non_rmos), capsize=5)
    axs[1].legend(mos_legend, bbox_to_anchor=(1, 0.7))

    plt.tight_layout()
    plt.show()

    # For Perturbation
    xcom_legend = [
        "xCoM",
        "xCoM peaks",
        "xCoM troughs",
        "L COP",
        "R COP",
    ]
    mos_legend = ["L MoS", "R MoS"]
    fig, axs = plt.subplots(2)
    fig.suptitle("Measurement of Stability WT M1")

    # For plotting figure demonstrating how calculation was done
    axs[0].set_title("How MoS is Derived")
    # axs[0].plot(wt1per_xcom)
    # axs[0].plot(wt1per_xcom_peaks, wt1per_xcom[wt1per_xcom_peaks], "^")
    # axs[0].plot(wt1per_xcom_troughs, wt1per_xcom[wt1per_xcom_troughs], "v")
    # axs[0].plot(wt1per_leftcop)
    axs[0].plot(wt1per_rightcop)
    # axs[0].legend(xcom_legend, bbox_to_anchor=(1, 0.7))

    # # Looking at result
    # axs[1].set_title("MoS Result")
    # axs[1].bar(0, np.mean(wt1per_lmos), yerr=np.std(wt1per_lmos), capsize=5)
    # axs[1].bar(1, np.mean(wt1per_rmos), yerr=np.std(wt1per_rmos), capsize=5)
    # axs[1].legend(mos_legend, bbox_to_anchor=(1, 0.7))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
