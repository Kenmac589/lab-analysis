import csv
import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp


def read_all_csv(directory_path):
    data_dict = {}  # Initialize an empty dictionary to store the data

    if not os.path.isdir(directory_path):
        print(f"{directory_path} is not a valid directory.")
        return

    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            data = pd.read_csv(file_path)
            data_dict[filename] = data

    return data_dict


def step_duration(input_dataframe):
    """
    @param: inpo
    """
    # Define the value and column to search for
    value_to_find = 1
    column_to_search = "45 sw onset"
    column_for_time = "Time"
    column_for_treadmill = "2 Trdml"

    # Store time values and treadmill speed when the specified value is found
    time_values = []
    treadmill_speed = []

    # Iterate through the DataFrame and process matches
    for index, row in input_dataframe.iterrows():
        if row[column_to_search] == value_to_find:
            time_value = row[column_for_time]
            time_values.append(time_value)
            treadmill_value = row[column_for_treadmill]
            treadmill_speed.append(treadmill_value)

    # Calculate the differences between consecutive time values
    time_differences = []
    for i in range(len(time_values)):
        time_diff = time_values[i] - time_values[i - 1]
        time_differences.append(time_diff)

    # Finding the average value for the list
    time_differences_array = np.array(time_differences)
    treadmill_speed_array = np.array(treadmill_speed)

    # Creating masks to filter any values above 1 as this would be between distinct recordings
    recording_cutoff_high = 0.6
    recording_cutoff_low = 0.000
    cutoff_high = time_differences_array <= recording_cutoff_high
    cutoff_low = time_differences_array >= recording_cutoff_low
    combined_filter = np.logical_and(cutoff_low, cutoff_high)

    # Applying the filter to the arrays
    adjusted_time_differences = time_differences_array[combined_filter]
    adjusted_treadmill_speeds = treadmill_speed_array[combined_filter]
    adj_time_xaxis = np.arange(0, len(adjusted_time_differences))

    # Finding average step cylce for this length
    average_step_difference = np.mean(adjusted_time_differences)

    return adjusted_time_differences, adjusted_treadmill_speeds


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


def extract_cycles(input_dataframe, swonset_channel="44 sw onset"):
    """Get cycle periods
    @param input_dataframe: spike file input as *.csv
    @param swonset_channel: the channel with swing onsets

    @return step_cycles: the step cycles
    """
    # Define the value and column to search for
    value_to_find = 1
    column_to_search = swonset_channel
    column_for_time = "Time"
    column_for_treadmill = "2 Trdml"

    # Store time values and treadmill speed when the specified value is found
    time_values = []
    treadmill_speed = []

    # Iterate through the DataFrame and process matches
    for index, row in input_dataframe.iterrows():
        if row[column_to_search] == value_to_find:
            time_value = row[column_for_time]
            time_values.append(time_value)
            treadmill_value = row[column_for_treadmill]
            treadmill_speed.append(treadmill_value)

    # Calculate the differences between consecutive time values
    time_differences = []
    for i in range(len(time_values)):
        time_diff = time_values[i] - time_values[i - 1]
        time_differences.append(time_diff)

    # Finding the average value for the list
    time_differences_array = np.array(time_differences)

    # Creating masks to filter any values above 1 as this would be between distinct recordings
    recording_cutoff_high = 0.6
    recording_cutoff_low = 0.000
    cutoff_high = time_differences_array <= recording_cutoff_high
    cutoff_low = time_differences_array >= recording_cutoff_low
    combined_filter = np.logical_and(cutoff_low, cutoff_high)

    # Applying the filter to the arrays
    step_cycles = time_differences_array[combined_filter]

    return step_cycles


# TODO: Create method for approximating swing onset for DLCLive
def swingon_estim(input_dataframe, toey="24 toey (cm)"):
    """Full width half maxiumum calculation
    Currently a work in progress
    @param: motor_p_full_full: full length numpy array of selected motor
    primitives

    @return: mean_fwhm: Mean value for the width of the primitives
    """

    # Save
    toey_values = input_dataframe[toey].values

    for i in range(number_cycles):
        current_primitive = motor_p_full[i * 200 : (i + 1) * 200, synergy_selection - 1]

        # Find peaks
        peaks, properties = sp.signal.find_peaks(
            current_primitive, distance=40, width=2
        )
        max_ind = np.argmax(peaks)
        # min_ind = np.argmin(mcurrent_primitive[0:max_ind])

        # half_width_height = (mcurrent_primitive[max_ind] - mcurrent_primitive[min_ind]) / 2

        # print("Manually Calculated", half_width_height)
        max_width = properties["widths"][max_ind]
        fwhl.append(max_width)
        # fwhl_start = properties["left_ips"][max_ind]
        # fwhl_stop = properties["right_ips"][max_ind]
        # half_width_height = properties["width_heights"][max_ind]

        print("Scipy calculated", properties["widths"][max_ind])
        # print(peaks[max_ind])
    fwhl = np.asarray(fwhl)

    return fwhl


def stance_duration(
    input_dataframe, swonset_channel="44 sw onset", swoffset_channel="45 sw offset"
):
    """Stance duration during step cycle
    @param input_dataframe: spike file input as *.csv
    @param swonset_channel: the channel with swing onsets
    @param swoffset_channel: the channel with swing offsets

    @return stance_duration_lengths: How long each stance duration is
    @return stance_duration_timings: List of timings where stance duration begins
    """

    # Define the value and column to search for
    value_to_find = 1
    stance_begin = swoffset_channel
    # stance_end = swonset_channel
    # column_to_search = swonset_channel
    column_for_time = "Time"
    # column_for_treadmill = "2 Trdml"

    # Store time values and treadmill speed when the specified value is found
    time_values = []

    # Find the first stance phase to start tracking time duration
    first_stance = (
        input_dataframe[stance_begin]
        .loc[input_dataframe[stance_begin] == value_to_find]
        .index[0]
    )

    # Iterate through the DataFrame and process matches
    for index, row in input_dataframe.iloc[first_stance:].iterrows():
        if row[swoffset_channel] == value_to_find:
            # print("swoff found at", row[column_for_time])
            time_value = row[column_for_time]
            time_values.append(time_value)
        elif row[swonset_channel] == value_to_find:
            # print("swon found at", row[column_for_time])
            time_value = row[column_for_time]
            time_values.append(time_value)
            # treadmill_value = row[column_for_treadmill]
            # treadmill_speed.append(treadmill_value)

    # Calculate the differences between consecutive time values
    time_differences = []
    for i in range(len(time_values)):
        time_diff = time_values[i] - time_values[i - 1]
        time_differences.append(time_diff)

    # Finding the average value for the list
    time_differences_array = np.array(time_differences)

    # Creating masks to filter any values above 1 as this would be between distinct recordings
    recording_cutoff_high = 0.6
    recording_cutoff_low = 0.000
    cutoff_high = time_differences_array <= recording_cutoff_high
    cutoff_low = time_differences_array >= recording_cutoff_low
    combined_filter = np.logical_and(cutoff_low, cutoff_high)
    # Applying the filter to the arrays
    stance_duration_lengths = time_differences_array[combined_filter]

    stance_duration_timings = time_values

    return stance_duration_lengths, stance_duration_timings


def weighted_slope(input_dataframe, p, comy="37 CoMy (cm)"):
    data = input_dataframe[comy].values

    n = len(data)
    slope = [0] * n  # initialize with zeros

    for i in range(p, n - p):
        past = data[i - p : i]
        future = data[i + 1 : i + p + 1]

        # calculate means of past and future points
        mean_past = np.mean(past)
        mean_future = np.mean(future)

        # update slope at time i using the calculated means
        slope[i] = (mean_future - mean_past) / 2

    slope = np.array(slope)

    return slope


def median_filter(arr, k):
    """
    :param arr: input numpy array
    :param k: is the size of the window you want to slide over the array (kernel).

    :return filtarr: An array of the same length where each element is the median of
    a window centered around the index in the array.
    """
    # Initialize output array
    result = []

    # Iterate over every index in arr
    for i in range(len(arr)):
        if i < (k // 2) or i > len(arr) - (k // 2) - 1:
            # Add a placeholder for the indices before k//2 and after length of array - k//2 - 1
            result.append(np.nan)
        else:
            # Calculate median within window and append to result list
            result.append(np.median(arr[i - (k // 2) : i + (k // 2) + 1]))

    filter = np.isnan(comy_values)
    filtarr = np.asarray(result)
    filtarr = filtarr[~filter]

    return filtarr


def slope(data, p):
    slopes = []

    # Iterate through each element of the data (except last 'p' elements)
    for i in range(len(data) - p - 1):
        # Calculating means using equal weighting
        mean_before = sum(data[i : i + p]) / p if p > 0 else 0
        mean_after = sum(data[i + 1 : i + p + 1]) / p if p > 0 else 0

        # Calculating slope using the line equation (y2-y1)/(x2-x1)
        slope = (mean_before - mean_after) / (p + p)

        slopes.append(slope)

    return np.asarray(slopes)


def fir_filter(data, taps):
    """
    data : input signal
    taps : number of filter taps
    """

    # Define the FIR filter using a low-pass prototype.
    taps = min(len(data), taps)
    cutoff_freq = 0.3  # Cutoff frequency is set to 30% of the Nyquist rate.
    # This means that only frequencies below this will pass through the filter,
    # and all others will be attenuated.

    nyquist = len(data) // 2
    freqs = np.arange(taps) / taps
    gains = 0.5 - 0.5 * np.cos(2 * np.pi * (freqs - cutoff_freq))

    # Apply the filter using a convolution, which is equivalent to multiplication in the frequency domain.
    filtered = sp.signal.lfilter([1], gains, data)

    return filtered


# TODO: Need visit documentation to understand how to get slope
def spike_slope(input_dataframe, time_constant, comy="37 CoMy (cm)"):

    # Converting time constant to meaningful indices
    time_factor = int(time_constant / 2)

    # Bring in data and convert to numpy array
    comy_values = input_dataframe[comy].values

    # Remove missing values
    # filter = comy_values[np.logical_not(np.isnan(comy_values))]
    filter = np.isnan(comy_values)
    comy_values = comy_values[~filter]

    # Calculate slope based on spike 2 specifications

    slope = []

    for i in range(len(comy_values)):
        if i <= 5:
            values_to_consider = np.array(comy_values[0:time_factor])
            current_slope = np.mean(values_to_consider)
            slope.append(current_slope)
        else:
            previous_values = np.array(comy_values[i - time_factor : i])
            next_values = np.array(comy_values[i : i + time_factor])
            current_slope = (np.mean(previous_values) + np.mean(next_values)) / 2
            slope.append(current_slope)

    return slope


# TODO: Center of pressure calculation needs to get done
def copressure(input_dataframe, ds_channel, hl_channel, fl_channel):
    """Calculation for center of pressure
    :param input_dataframe: spike file input as *.csv

    """
    input_dataframe_subset = input_dataframe.loc[
        :, ["Time", ds_channel, hl_channel, fl_channel]
    ]

    print(input_dataframe_subset)


def step_width(
    input_dataframe: pd.DataFrame,
    rl_stance: Union[np.ndarray, list],
    ll_stance: Union[np.ndarray, list],
    rl_y: str,
    ll_y: str,
) -> np.array:
    """Step width during step cycle
    :param input_dataframe: spike file input as *.csv
    :param rl_stance: when stance begins for the right limb
    :param ll_stance: when stance begins for the left limb
    :param rl_y: spike channel with y coordinate for the right limb
    :param ll_y: spike channel with y coordinate for the right limb

    :return step_widths: array of step width values for each step cycle
    """

    # Filtering whole dataframe down to values we are considering
    input_dataframe_subset = input_dataframe.loc[:, ["Time", rl_y, ll_y]]
    input_dataframe_subset = input_dataframe_subset.set_index("Time")

    # Grabbing analogous values from
    ll_step_placement = input_dataframe_subset.loc[ll_stance, :][ll_y].values
    rl_step_placement = input_dataframe_subset.loc[rl_stance, :][rl_y].values

    # Dealing with possible unequal amount of recorded swoffsets for each limb
    comparable_steps = 0
    if rl_step_placement.shape[0] >= ll_step_placement.shape[0]:
        comparable_steps = ll_step_placement.shape[0]
    else:
        comparable_steps = rl_step_placement.shape[0]

    step_widths = []

    # Compare step widths for each step
    for i in range(comparable_steps):
        new_width = np.abs(rl_step_placement[i] - ll_step_placement[i])
        step_widths.append(new_width)

    step_widths = np.asarray(step_widths)

    return step_widths


def hip_height(input_dataframe, toey="24 toey (cm)", hipy="16 Hipy (cm)", manual=False):
    """Approximates Hip Height
    :param input_dataframe: spike file input as *.csv
    :param toey: spike channel with y coordinate for the toe
    :param hipy: spike channel with y coordinate for the hip

    :return hip_height: returns hip height in meters (cm)
    """

    # Bringing in the values for toey and hipy
    toey_values = input_dataframe[toey].values
    hipy_values = input_dataframe[hipy].values

    # Remove missing values
    toey_values = toey_values[np.logical_not(np.isnan(toey_values))]
    hipy_values = hipy_values[np.logical_not(np.isnan(hipy_values))]

    # Either manually mark regions foot is on the ground or go with proxy
    if manual is False:

        # Getting lower quartile value of toey as proxy for the ground
        toey_lowerq = np.percentile(toey_values, q=25)
        average_hip_value = np.mean(hipy_values)
        hip_height = average_hip_value - toey_lowerq

    elif manual is True:
        # Selection of regions foot would be on the ground
        # on_ground_regions, _ = manual_marks(
        #     toey, title="Select Regions foot is on the ground"
        # )
        on_ground_regions = [0, 1, 2, 3, 4, 5]
        toe_to_consider = np.array([])
        hip_to_consider = np.array([])
        stance_begin = on_ground_regions[0::2]
        swing_begin = on_ground_regions[1::2]

        for i in range(len(stance_begin)):
            # Get regions to consider
            begin = stance_begin[i]
            end = swing_begin[i]

            relevant_toe = toey_values[begin:end]
            relevant_hip = hipy_values[begin:end]

            toe_to_consider = np.append(toe_to_consider, relevant_toe)
            hip_to_consider = np.append(hip_to_consider, relevant_hip)

    else:
        print("The `manual` variable must be a boolean")

    return hip_height


def froud_number(
    input_dataframe, trdm="2 Trdml", toey="24 toey (cm)", hipy="16 Hipy (cm)"
):
    # Need to get hip height and convert into centimeters
    hip = hip_height(input_dataframe, toey, hipy) / 100

    # Getting average treadmill speed
    treadmill_speed = input_dataframe[trdm].values
    avg_treadmill_speed = np.round(np.mean(treadmill_speed), 2)

    # Calculating number
    froud_number = np.power(avg_treadmill_speed, 2) / (9.81 * hip)

    return froud_number


# NOTE: Depending on completion of `spike_slope`
def xcom(input_dataframe, hip_height, comy="37 CoMy (cm)"):

    # Bring in data
    comy_values = input_dataframe[comy].values
    comy_values = comy_values[np.logical_not(np.isnan(comy_values))]

    # Getting slope of values
    vcom = slope(comy_values, 6)

    # Get xCoM in (cm)
    xcom = comy_values + (vcom / np.sqrt(981 / hip_height))

    # print("comy", comy_values.size)
    # print("vcom", vcom.size)
    # print("xcom", xcom)

    # x_axis = np.arange(len(comy_values))

    # Testing output
    # rbf = sp.interpolate.Rbf(x_axis, vcom, function="thin_plate", smooth=2)
    # xnew = np.linspace(x_axis.min(), x_axis.max(), num=100, endpoint=True)
    # ynew = rbf(xnew)

    # fig, axs = plt.subplots(4, 1, layout="constrained")
    # axs[0].set_title("CoMy")
    # axs[0].legend(loc="best")
    # axs[0].plot(x_axis, comy_values)
    # axs[1].set_title("vCoM")
    # axs[1].plot(x_axis, vcom)
    # axs[2].set_title("Radial basis funtion interpolation of vCoM")
    # axs[2].plot(xnew, ynew)
    # axs[0].plot(x_axis, xcom)

    # plt.show()

    return xcom


def mos(xcom, leftcop, rightcop, manual_peaks=False, width_threshold=40):

    # Remove periods where it is not present or not valid
    left_band = np.percentile(xcom, q=50)
    right_band = 3
    rightcop[rightcop < right_band] = np.nan
    leftcop[leftcop < left_band] = np.nan

    # Optional manual point selection
    if manual_peaks is False:
        # Getting peaks and troughs
        xcom_peaks, _ = sp.signal.find_peaks(xcom, width=width_threshold)
        xcom_troughs, _ = sp.signal.find_peaks(-xcom, width=width_threshold)
    elif manual_peaks is True:
        xcom_peaks, _ = manual_marks(xcom, title="Select Peaks")
        xcom_troughs, _ = manual_marks(xcom, title="Select Troughs")
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


def cycle_period_summary(directory_path):

    trial_list = read_all_csv(directory_path)

    cycle_results = {}
    for key in trial_list:
        cycle_results[key] = None

    # Now, you can access the data from each file like this:
    for filename, data in trial_list.items():
        step_duration_array, treadmill_speed = step_duration(data)
        cycle_results[filename] = np.mean(step_duration_array), np.mean(treadmill_speed)

    # Saving results to csv
    cycle_results_csv = "cycle_analysis.csv"

    with open(cycle_results_csv, "w", newline="") as file:
        writer = csv.writer(file)

        # Write the header row (optional)
        writer.writerow(["Data Point", "Mean", "Standard Deviation"])

        # Write data from the dictionary
        for key, (mean, std_dev) in cycle_results.items():
            writer.writerow([key, mean, std_dev])

    print(f"Data has been saved to {cycle_results_csv}")


# Main Code Body
def main():

    # Test for speed of step width
    # wt1nondf = pd.read_csv("./wt_1_non-perturbation.csv")
    # wt2nondf = pd.read_csv("./wt-2-non-perturbation-all.txt", delimiter=",", header=0)
    # wt2perdf = pd.read_csv("./wt-2-perturbation-all.txt", delimiter=",", header=0)
    # wt4nondf = pd.read_csv("./wt_4_non-perturbation.csv")
    # wt4perdf = pd.read_csv("./wt_4_perturbation.csv")
    # wt5nondf = pd.read_csv("./wt-5-non-perturbation-all.txt", delimiter=",", header=0)
    # wt5perdf = pd.read_csv("./wt-5-perturbation-all.txt", delimiter=",", header=0)

    # For Egr3 KO's
    egr3_6nondf = pd.read_csv(
        "./egr3-6-non-perturbation-all.txt", delimiter=",", header=0
    )
    egr3_7nondf = pd.read_csv(
        "./egr3-7-non-perturbation-all.txt", delimiter=",", header=0
    )
    egr3_8nondf = pd.read_csv(
        "./egr3-8-non-perturbation-all.txt", delimiter=",", header=0
    )
    egr3_9nondf = pd.read_csv(
        "./egr3-9-non-perturbation-all.txt", delimiter=",", header=0
    )
    egr3_10nondf = pd.read_csv(
        "./egr3-10-non-perturbation-all.txt", delimiter=",", header=0
    )
    # wt1xcom = pd.read_csv("./wt-1_non-perturbation-cop.txt")
    # spike_com = wt1xcom["37a CoMy (cm)"].values
    # spike_xcom = wt1xcom["67 xCoM"].values
    # Getting stance duration for all 4 limbs
    # lhl_st_lengths, lhl_st_timings = stance_duration(
    #     wt4nondf, swonset_channel="57 lHL swon", swoffset_channel="58 lHL swoff"
    # )
    # lfl_st_lengths, lfl_st_timings = stance_duration(
    #     wt4nondf, swonset_channel="53 lFL swon", swoffset_channel="54 lFL swoff"
    # )
    # rhl_st_lengths, rhl_st_timings = stance_duration(
    #     wt4nondf, swonset_channel="55 rHL swon", swoffset_channel="56 rHL swoff"
    # )
    # rfl_st_lengths, rfl_st_timings = stance_duration(wt4nondf)
    # print(f"Right stance duration {rfl_st_lengths}\n")
    # print(f"Right stance phase beginning {rfl_st_timings}\n")

    # # For forelimb
    # fl_step_widths = step_width(
    #     wt4nondf, rfl_st_timings, lfl_st_timings, rl_y="35 FRy (cm)", ll_y="33 FLy (cm)"
    # )
    # print(fl_step_widths)
    # hl_step_widths = step_width(
    #     wt4nondf, rhl_st_timings, lhl_st_timings, rl_y="30 HRy (cm)", ll_y="28 HLy (cm)"
    # )

    # print(
    #     copressure(
    #         wt1nondf, ds_channel="59 Left DS", hl_channel="28 HLy", fl_channel="33 FLy"
    #     )
    # )

    # Getting hip heights
    # wt2non_hip_h = hip_height(wt2nondf, toey="24 toey (cm)", hipy="16 Hipy (cm)")
    # wt2per_hip_h = hip_height(wt2perdf, toey="24 toey (cm)", hipy="16 Hipy (cm)")
    # wt4non_hip_h = hip_height(wt4nondf, toey="24 toey (cm)", hipy="16 Hipy (cm)")
    # wt4non_hip_h = hip_height(wt4nondf, toey="24 toey (cm)", hipy="16 Hipy (cm)")
    # wt4per_hip_h = hip_height(wt4perdf, toey="24 toey (cm)", hipy="16 Hipy (cm)")
    # wt5non_hip_h = hip_height(wt5nondf, toey="24 toey (cm)", hipy="16 Hipy (cm)")
    # wt5per_hip_h = hip_height(wt5perdf, toey="24 toey (cm)", hipy="16 Hipy (cm)")

    # egr3_6non_hip_h = hip_height(egr3_6nondf, toey="24 toey (cm)", hipy="16 Hipy (cm)")
    # egr3_7non_hip_h = hip_height(egr3_7nondf, toey="24 toey (cm)", hipy="16 Hipy (cm)")
    # egr3_8non_hip_h = hip_height(egr3_8nondf, toey="24 toey (cm)", hipy="16 Hipy (cm)")
    # egr3_9non_hip_h = hip_height(egr3_9nondf, toey="24 toey (cm)", hipy="16 Hipy (cm)")
    egr3_10non_hip_h = hip_height(
        egr3_10nondf, toey="24 toey (cm)", hipy="16 Hipy (cm)"
    )

    # print(f"Egr3 M6 Hip {egr3_6non_hip_h}")
    # print(f"Egr3 M7 Hip {egr3_7non_hip_h}")
    # print(f"Egr3 M8 Hip {egr3_8non_hip_h}")
    # print(f"Egr3 M9 Hip {egr3_9non_hip_h}")
    # print(f"Egr3 M10 Hip {egr3_10non_hip_h}")

    # Working through xcom caluclation to be better
    # com = wt1nondf["37 CoMy"].values
    # func_test = slope(com, 1)
    # vcom = weighted_slope(wt1nondf, 6, comy="37 CoMy")
    # # xcom_wt1 = xcom(wt1nondf, hip_h, comy="37 CoMy")
    #
    # # print(vcom)
    # fig, axs = plt.subplots(3)
    # legend = ["CoMy", "xCoM"]
    # axs[0].plot(com)
    # # axs[0].plot(xcom_wt1)
    # axs[0].legend(legend)
    # axs[1].plot(spike_xcom)
    # axs[1].plot(spike_com)
    # axs[1].legend(legend)
    # axs[2].plot(func_test)
    # plt.show()


if __name__ == "__main__":
    main()
# %%
