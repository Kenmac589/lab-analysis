import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

    # Grabbing individual channels
    xcom = wt1nondf["67 xCoM"].to_numpy(dtype=float)
    leftcop = wt1nondf["v5 L COP"].to_numpy(dtype=float)
    rightcop = wt1nondf["v3 R COP"].to_numpy(dtype=float)

    # Remove periods where it is not present
    leftcop[leftcop == 0.0] = np.nan
    rightcop[rightcop == 0.0] = np.nan

    lds_timings = double_support_timings(wt1nondf, ds_channel="59 Left DS")
    rds_timings = double_support_timings(wt1nondf, ds_channel="60 Right DS")
    # print(lds_timings)
    # "v6 L Mos" and "v4 R Mos"

    filtered_lmos = cop_filter(wt1nondf, lds_timings, cop_channel="67 xCoM")
    filtered_rmos = cop_filter(wt1nondf, rds_timings, cop_channel="67 xCoM")

    # Plotting
    fig, axs = plt.subplots(3)
    fig.suptitle("Measurement of Dynamic Stability (MoS)")
    mos_legend = [
        "L xCoM timings",
        "R xCoM timings",
    ]
    ds_legend = ["Left COP", "Right COP"]

    # Mos Related
    # axs[0].plot(wt1nondf["Time"], xcom)
    axs[0].plot(wt1nondf["Time"], filtered_lmos)
    axs[0].plot(wt1nondf["Time"], filtered_rmos)
    axs[0].legend(mos_legend)

    # Double support
    axs[1].plot(wt1nondf["Time"], leftcop)
    axs[2].plot(wt1nondf["Time"], rightcop)
    # axs[1].legend(ds_legend)
    plt.show()


if __name__ == "__main__":
    main()
