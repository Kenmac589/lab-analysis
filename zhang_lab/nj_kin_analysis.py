import csv

import dlc2kinematics as dlck
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal


def frame_to_time(frame_index, fps=250):
    """Converts frames of video to time based on fps

    Parameters
    ----------
    frame_index:
        The index of the frame in question
    fps:
        The frames per second the video is shot at

    Returns
    -------
    time_seconds:
        Returns the time in seconds the frame is taken at

    """

    # Convert to miliseconds
    frame_mili = (frame_index / fps) * 1000
    # Convert to seconds
    time_seconds = frame_mili / 1000

    return time_seconds


def dlc_calibrate(df, bodyparts, scorer, calibration_markers):
    """
    Calibrate DLC output to physical distance via calibration marker estimation.
    This is redundant in ways as it's likely this will be loaded in for other things.

    Parameters
    ----------
    df: pandas.core.frame.Dataframe
        h5 file from DeepLabCut
    bodyparts: list
        This is output from the loaded dataset
    scorer: str
        This is output from the loaded dataset
    calibration_markers:
        List of names for calibration markers

    Returns
    -------
    calibration_factor:
        float value to convert given pixel values to (cm)
    """

    avg_cal_cords = {}

    # HACK: Hard coded to shit plz fix later
    # This is done explicitly based on how I label
    top_row = calibration_markers[0:3]
    bottom_row = calibration_markers[3:6]

    for i in range(len(calibration_markers)):
        # Loading in Respective Dataframe for each marker
        calib_shown = df[scorer][calibration_markers[i]]
        calib_shown = calib_shown.drop(columns=["likelihood"])

        # Get average position
        avg_cord_np = np.array([])
        avg_x, avg_y = calib_shown["x"].mean(), calib_shown["y"].mean()
        avg_cord_np = np.append(avg_cord_np, [avg_x, avg_y])
        entry = calibration_markers[i]
        avg_cal_cords[entry] = avg_cord_np

    # Getting average x movement factor (2 cm across)
    two_cm_difs = np.array([])
    for i in range(2):
        top_dif = avg_cal_cords[top_row[i + 1]][0] - avg_cal_cords[top_row[i]][0]
        bottom_dif = (
            avg_cal_cords[bottom_row[i + 1]][0] - avg_cal_cords[bottom_row[i]][0]
        )
        two_cm_difs = np.append(two_cm_difs, top_dif)
        two_cm_difs = np.append(two_cm_difs, bottom_dif)

    x_factor = np.mean(two_cm_difs) / 2
    # print(x_factor)

    # Getting average y movement factor (2.5 cm across)
    # Noting y values towards bottom of image are higher than the top
    two_p5_cm_difs = np.array([])
    for i in range(3):
        y_dif = avg_cal_cords[bottom_row[i]][1] - avg_cal_cords[top_row[i]][1]
        two_p5_cm_difs = np.append(two_p5_cm_difs, y_dif)

    y_factor = np.mean(two_p5_cm_difs) / 2.5
    # print(y_factor)

    calibration_factor = np.mean([x_factor, y_factor])
    # print(calibration_factor)

    return calibration_factor


def manual_marks(related_trace, title="Select Points"):
    """Manually annotate points of interest on a given trace

    Parameters
    ----------
    related_trace:
        Trace you want to annotate

    Returns
    -------
    manual_marks_x:
        array of indices to approx desired value in original trace
    manual_marks_y:
        array of selected values
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


def analysis_reg_sel(mirror_y, com_y):

    time = np.arange(0, len(com_y), 1)

    # Open interface with trace
    plt.plot(time, mirror_y, label="Mirror Y")
    plt.plot(time, com_y, label="CoM Y")
    plt.title("Label Start and Stop of Desired Regions")

    # Go through and label regions desired
    manual_marks_pair = plt.ginput(0, 0)

    # Store x coordinates as rounded off ints to be used as indices
    manual_marks_x = np.asarray(list(map(lambda x: x[0], manual_marks_pair)))
    manual_marks_x = manual_marks_x.astype(np.int32)

    plt.show()

    region_start = manual_marks_x[0]
    region_stop = manual_marks_x[1]

    return region_start, region_stop


def mark_process(df, scorer, marker, cord, calib=1, smooth_wind=40):
    """Extracts out and smoothens a marker coordinate to an numpy array.

    Parameters
    ----------
    df: pandas.core.frame.Dataframe
        input h5 data file to get marker from
    scorer: str
        Who scored the that trained the model, which analyzed the videos
    marker: str
        Name of marker you want to extract
    cord: str
        Whether you want the `x` or `y` coordinate of the marker
    calib: numpy.float64
        Calibration factor calculated from `dlc_calibrate`
    smooth_wind: int, default=`40`
        Just the smoothening window of `scipy.signal.savgol_filter`

    Returns
    -------
    marker_np: numpy.ndarray
        Smoothened 1D numpy array of the given marker
    """

    mark_df = df[scorer][marker]

    # Get single dimension out and calibrate to (cm)
    marker_np = pd.array(mark_df[cord])
    marker_np = marker_np / calib

    # Smoothen
    marker_np = signal.savgol_filter(marker_np, smooth_wind, 3)

    return marker_np


def swing_estimation(foot_cord, manual=False, width_threshold=20):
    """This approximates swing onset and offset from kinematic data

    Parameters
    ----------
    foot_cord:
        Exported channels from spike most importantly the x values for a channel

    Returns
    -------
    swing_onset:
        A list of indices where swing onset occurs
    swing_offset:
        A list of indices where swing offset occurs
    """

    if manual is False:
        # Getting peaks and troughs
        swing_offset, _ = signal.find_peaks(foot_cord, width=width_threshold)
        swing_onset, _ = signal.find_peaks(-foot_cord, width=width_threshold)
    elif manual is True:
        swing_offset, _ = manual_marks(foot_cord, title="Select Swing offsets")
        swing_onset, _ = manual_marks(-foot_cord, title="Select Swing onsets")
    else:
        print("The `manual` variable must be a boolean")

    return swing_onset, swing_offset


def swing_timings(foot_cord, manual=False, width_threshold=20):
    """This approximates swing onset and offset from kinematic data

    Parameters
    ----------
    foot_cord:
        Exported channels from spike most importantly the x values for a channel

    Returns
    -------
    swon_timings:
        A list of timings in seconds where swing onset occurs
    swoff_timings:
        A list of timings in seconds where swing offset occurs
    """

    if manual is False:
        # Getting peaks and troughs
        swing_offset, _ = signal.find_peaks(foot_cord, width=width_threshold)
        swing_onset, _ = signal.find_peaks(-foot_cord, width=width_threshold)
    elif manual is True:
        swing_offset, _ = manual_marks(foot_cord, title="Select Swing offsets")
        swing_onset, _ = manual_marks(-foot_cord, title="Select Swing onsets")
    else:
        print("The `manual` variable must be a boolean")

    # Converting indices to time
    swon_timings = frame_to_time(swing_onset, fps=250)
    swoff_timings = frame_to_time(swing_offset, fps=250)

    return swon_timings, swoff_timings


def step_cycle_est(foot_cord, manual=False, width_threshold=20):
    """This approximates swing onset and offset from kinematic data
    Parameters
    ----------
    input_dataframe:
        Exported channels from spike most importantly the x values for a channel

    Returns
    -------
    cycle_durations:
        A numpy array with the duration of each cycle
    average_step:
        A list of indices where swing offset occurs
    """

    # Calculating swing estimations
    swing_onset, _ = swing_estimation(
        foot_cord, manual=manual, width_threshold=width_threshold
    )

    # Converting Output to time in seconds
    time_conversion = np.vectorize(frame_to_time)
    onset_timing = time_conversion(swing_onset)

    cycle_durations = np.array([])
    for i in range(len(onset_timing) - 1):
        time_diff = onset_timing[i + 1] - onset_timing[i]
        cycle_durations = np.append(cycle_durations, time_diff)

    return cycle_durations


# Custom median filter from
def median_filter(arr, k):
    """Median filter attempting to be faithful to Spike2's channel process

    Parameters
    ----------
    arr:
        input numpy array
    k:
        The size of the window you want to slide over the array.
        This is also considered the kernel

    Returns
    -------
    result:
        A numpy array of the same length where each element is the median of
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

    return np.array(result)


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


def spike_slope(comy, p):
    """Getting slope of curve attempting to be faithful to Spike2's channel process

    :param comy: numpy array of the y coordinate of the center of mass
    :param p: How many values you want in either direction to be included


    """

    n = len(comy)
    slope = [0] * n  # initialize with zeros

    for i in range(p, n - p):
        past = comy[i - p : i]
        future = comy[i + 1 : i + p + 1]

        # calculate means of past and future points
        mean_past = np.mean(past)
        mean_future = np.mean(future)

        # update slope at time i using the calculated means
        slope[i] = (mean_future - mean_past) / 2

    slope = np.array(slope)

    return slope


def hip_height(toey_values, hipy_values, manual=False):
    """
    Approximates Hip Height.

    Args:
    :param toey_values: numpy array of values for y coordinate of toe marker
    :param hipy_values: numpy array of values for y coordinate of hip marker
    :param manual: (Boolean) whether to manually label regions where foot is on ground

    Returns
    -------
    hip_height: returns hip height in meters (cm)
    """

    # Either manually mark regions foot is on the ground or go with proxy
    if manual is False:

        # Getting lower quartile value of toey as proxy for the ground
        toey_lowerq = np.percentile(toey_values, q=25)
        average_hip_value = np.mean(hipy_values)
        hip_height = toey_lowerq - average_hip_value

    elif manual is True:
        # Selection of regions foot would be on the ground
        on_ground_regions, _ = manual_marks(
            toey_values, title="Select Regions foot is on the ground"
        )

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

        # Calculate hip height from filtered regions
        average_hip_value = np.mean(hip_to_consider)
        average_toe_value = np.mean(toe_to_consider)
        hip_height = average_toe_value - average_hip_value

    else:
        print("The `manual` variable must be a boolean")

    return hip_height


def step_width_est(
    rl_x: np.array,
    ll_x: np.array,
    rl_y: np.array,
    ll_y: np.array,
):
    """Step width during step cycle

    Parameters
    ----------
    rl_x: numpy.ndarray
        1D array of the x coordinate for the right limb
    ll_x: numpy.ndarray
        1D array of the x coordinate for the left limb
    rl_y: numpy.ndarray
        1D array of the y coordinate for the right limb
    ll_y: numpy.ndarray
        1D array of the y coordinate for the left limb

    Returns
    -------
    step_widths: numpy.ndarray
        numpy array of step width values for each step cycle
    """

    # Filtering whole dataframe down to values we are considering
    rl_y_cords = rl_y
    ll_y_cords = ll_y

    _, rl_swoff = swing_estimation(rl_x)
    _, ll_swoff = swing_estimation(ll_x)

    rl_step_placement = rl_y_cords[rl_swoff]
    ll_step_placement = ll_y_cords[ll_swoff]

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


def limb_measurements(
    input_skeleton,
    skeleton_list,
    calibration,
    save_as_csv=False,
    csv_filename="./tmp-limb_measurmets.csv",
):
    """Estimates lengths of limbs coordinates based on skeleton reconstruction

    Parameters
    ----------
    input_skeleton:
        raw dataframe from spike no cleaning necessary
    skeleton_list:
        list of components in skeleton to look for in dataframe
    calibration:
        calibration factor calculated from recording itself

    Returns
    -------
    calibrated_measurements:
        Dictionary of the average distance for each joint
    """
    calibrated_measurments = {}

    input_skeleton = input_skeleton[skeleton_list[:]]
    sk_df = input_skeleton.drop([0])  # removes row saying length
    # print(sk_df)

    for key in skeleton_list:
        column_values = pd.array(sk_df[key], dtype=np.dtype(float)) / calibration
        calibrated_measurments[key] = np.mean(column_values)

    if save_as_csv is True:
        w = csv.writer(open(csv_filename, "w"))
        for key, val in calibrated_measurments.items():
            w.writerow([key, val])

    return calibrated_measurments


# NOTE: Anyone at the Zhang lab using this script it's a hella work in progress
# If you have any questions email me at kenzie.mackinnon@dal.ca or just come talk
# to me. Will be able to do most of the kinematic stuff you guys to but unless there
# is a dot for center or mass then alot of the stability functions won't work.
def main():

    # NOTE: Very important this is checked before running
    video = 3
    mouse = 458
    manual = False  # Whether to annotate dots manually
    save_auto = False  # Whether to save plot generated automatically
    show_plots = True  # Whether or not you want the plots and data saved
    calib_factor = 1

    # Settings before running initial workup from DeepLabCut
    figure_title = f"Step Cycles for NJ{mouse}-vid-{video}"
    figure_filename = f"./NJ{mouse}/NJ{mouse}-swing_timings/NJ{mouse}-{video}.pdf"
    step_cycles_filename = (
        f"./NJ{mouse}/NJ{mouse}-swing_timings/NJ{mouse}-step-cycles-{video}.csv"
    )
    swing_times_file = (
        f"./NJ{mouse}/NJ{mouse}-swing_timings/NJ{mouse}-swing_times-{video}.csv"
    )

    # Loading in a dataset
    df, bodyparts, scorer = dlck.load_data(
        f"./NJ{mouse}/NJ{mouse}_lvl_075_15cms_left_{video}DLC_resnet101_V3OFFKAMay3shuffle1_1010000.h5"
    )

    # -------------------------------------------------------------------------

    # Convert calibration_factor to proper dt
    calib_factor = np.float64(calib_factor)

    # Grabbing marker data
    ftoex_np = mark_process(df, scorer, "fronttoe", "x", calib_factor)
    rflx_np = mark_process(df, scorer, "fr", "x", calib_factor)
    rhlx_np = mark_process(df, scorer, "hr", "x", calib_factor)
    lflx_np = mark_process(df, scorer, "fl", "x", calib_factor)
    lhlx_np = mark_process(df, scorer, "hl", "x", calib_factor)
    toey_np = mark_process(df, scorer, "toe", "y", calib_factor)
    hipy_np = mark_process(df, scorer, "hip", "y", calib_factor)
    rfly_np = mark_process(df, scorer, "fr", "y")
    rhly_np = mark_process(df, scorer, "hr", "y")
    lfly_np = mark_process(df, scorer, "fl", "y")
    lhly_np = mark_process(df, scorer, "hl", "y")

    # Getting a time adjusted array of equal length for time
    time = np.arange(0, len(ftoex_np), 1)
    time = frame_to_time(time, fps=250)

    # Step cycle Estimation
    step_cyc_durations = step_cycle_est(ftoex_np)
    ftoe_swing_onset, ftoe_swing_offset = swing_estimation(
        ftoex_np, manual=manual, width_threshold=15
    )

    ftoe_swon_times, ftoe_swoff_times = swing_timings(ftoex_np)
    rfl_swon_times, rfl_swoff_times = swing_timings(foot_cord=rflx_np)
    rhl_swon_times, rhl_swoff_times = swing_timings(foot_cord=rhlx_np)
    lfl_swon_times, lfl_swoff_times = swing_timings(foot_cord=lflx_np)
    lhl_swon_times, lhl_swoff_times = swing_timings(foot_cord=lhlx_np)
    ftoe_swon_times = pd.Series(ftoe_swon_times, name="Swing Onset")
    ftoe_swoff_times = pd.Series(ftoe_swoff_times, name="Swing Offset")

    swing_times = pd.DataFrame(
        {"Swing Onset": ftoe_swon_times, "Swing Offset": ftoe_swoff_times}
    )

    # Hip Height
    hip_h = hip_height(toey_values=toey_np, hipy_values=hipy_np)

    # Step Width's
    fl_stepw = step_width_est(rflx_np, lflx_np, rfly_np, lfly_np)
    hl_stepw = step_width_est(rhlx_np, lhlx_np, rhly_np, lhly_np)

    # Printing timings

    # Some of my default plotting parameters I like
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set(
        style="white",
        font_scale=1.3,
        palette="colorblind",
        rc=custom_params,
    )

    # Figure formatting
    fig = plt.figure(figsize=(19.2, 10.80))
    fig.tight_layout()
    axs = fig.subplot_mosaic(
        [
            ["toe_trace", "toe_trace", "toe_trace"],
            ["step_cycles", "hip_height", "step_width"],
        ]
    )

    # Plot Legend

    # For plotting figure demonstrating how swing estimation was done
    axs["toe_trace"].set_title(figure_title)
    axs["toe_trace"].plot(ftoex_np, label="Limb X cord")
    axs["toe_trace"].plot(
        ftoe_swing_offset, ftoex_np[ftoe_swing_offset], "^", label="Swing offset"
    )
    axs["toe_trace"].plot(
        ftoe_swing_onset, ftoex_np[ftoe_swing_onset], "v", label="Swing onset"
    )
    axs["toe_trace"].legend(bbox_to_anchor=(1, 0.7), loc="upper left")

    # For Step Cycles
    axs["step_cycles"].set_ylabel("Step Cycle Duration")
    axs["step_cycles"].bar(
        0, np.mean(step_cyc_durations), yerr=np.std(step_cyc_durations), capsize=4
    )

    # For Hip Height
    axs["hip_height"].set_ylabel("Hip Height")
    axs["hip_height"].bar(0, hip_h)
    axs["hip_height"].set_ylim(0, hip_h * 2)

    # For Step Width
    axs["step_width"].set_ylabel("Step Width")
    axs["step_width"].bar(
        0, np.mean(fl_stepw), yerr=np.std(fl_stepw), capsize=4, label="Forelimb"
    )
    axs["step_width"].bar(
        1, np.mean(hl_stepw), yerr=np.std(hl_stepw), capsize=4, label="Hindlimb"
    )
    axs["step_width"].legend(loc="best")

    # Saving Figure in same folder

    if manual is True:
        # Saving plot and results
        plt.savefig(figure_filename, dpi=300)
        np.savetxt(step_cycles_filename, step_cyc_durations, delimiter=",")
        swing_times.to_csv(swing_times_file, index=False)
        print("Kinematic results saved")
    elif manual is False and save_auto is True:
        # Saving plot and results
        plt.savefig(figure_filename, dpi=300)
        np.savetxt(step_cycles_filename, step_cyc_durations, delimiter=",")
        swing_times.to_csv(swing_times_file, index=False)
        print("Kinematic results saved")
    elif manual is False and save_auto is False and show_plots is True:
        print("Kinematic results not saved")
        plt.show()
    else:
        print("Kinematic results not saved")


if __name__ == "__main__":
    main()
