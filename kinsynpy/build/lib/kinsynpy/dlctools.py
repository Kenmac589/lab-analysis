import dlc2kinematics as dlck
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dlc2kinematics import Visualizer2D
from scipy import signal


def frame_to_time(frame_index, fps=500):

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
    df :
        h5 file from DeepLabCut
    bodyparts:
        This is output from the loaded dataset
    scorer:
        This is output from the loaded dataset
    calibration_markers:
        List of names for calibration markers

    Returns
    -------
    calibration_factor:
        float value to convert given pixel values to (cm)
    """

    avg_cal_cords = {}

    # TODO: Hard coded to shit plz fix later
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
    print(x_factor)

    # Getting average y movement factor (2.5 cm across)
    # Noting y values towards bottom of image are higher than the top
    two_p5_cm_difs = np.array([])
    for i in range(3):
        y_dif = avg_cal_cords[bottom_row[i]][1] - avg_cal_cords[top_row[i]][1]
        two_p5_cm_difs = np.append(two_p5_cm_difs, y_dif)

    y_factor = np.mean(two_p5_cm_difs) / 2.5
    print(y_factor)

    calibration_factor = np.mean([x_factor, y_factor])
    print(calibration_factor)

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


def swing_estimation(foot_cord, manual=False, width_threshold=40):
    """This approximates swing onset and offset from kinematic data
    :param : Exported channels from spike most importantly the x values for a channel

    :return swing_onset: A list of indices where swing onset occurs
    :return swing_offset: A list of indices where swing offet occurs
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


def step_cycle_est(foot_cord, manual=False, width_threshold=40):
    """This approximates swing onset and offset from kinematic data
    :param input_dataframe: Exported channels from spike most importantly the x values for a channel

    :return cycle_durations: A numpy array with the duration of each cycle
    :return average_step: A list of indices where swing offet occurs
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
    """
    :param arr: input numpy array
    :param k: is the size of the window you want to slide over the array.
    also considered the kernel

    :return : An array of the same length where each element is the median of
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
    """
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
    """Approximates Hip Height
    :param toey_values: numpy array of values for y coordinate of toe marker
    :param hipy_values: numpy array of values for y coordinate of hip marker
    :param manual: (Boolean) whether to manually label regions where foot is on ground

    :return hip_height: returns hip height in meters (cm)
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


def xcom(comy, vcom, hip_height):
    """
    Calculates extrapolated Center-of-mass (xCoM)

    Parameters
    ----------
    comy:
        numpy array of values for y coordinate of toe marker
    hip_height:
        average hip height

    Returns
    -------
    xcom:
        A 1-D array representing the extrapolated CoM in cm
    """

    # Get xCoM in (cm)
    xcom = comy + vcom / np.sqrt(9.81 / hip_height)

    return xcom


def double_support(fl_x, hl_x, manual_analysis=False, filt_window=40):
    """Finds double support phases from forlimb movement
    :param fl_x: forelimb x coordinate
    :param hl_x: hindlimb x coordinate
    :param manual_analysis: Will allow for manual annotation of traces to pick phases.
    :param filt_window: Smoothening factor for traces.

    :return ds_phases: Same length 1-D array as input either 1 or np.nan values indicating phases
    """

    # Smoothen Traces
    fl_x = signal.savgol_filter(fl_x, filt_window, 3)
    hl_x = signal.savgol_filter(hl_x, filt_window, 3)

    fl_swon, fl_swoff = swing_estimation(fl_x, manual=False)
    hl_swon, hl_swoff = swing_estimation(fl_x, manual=False)

    return ds_phases


def cop(fl_y, hl_y):

    return (fl_y + hl_y) / 2


def mos_marks(related_trace, leftcop, rightcop, title="Select Points"):
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
    # plt.plot(leftcop)
    # plt.plot(rightcop)
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
    # left_band = np.percentile(xcom, q=50)
    rightcop = np.where(rightcop == 0.0, np.nan, rightcop)
    leftcop = np.where(leftcop == 0.0, np.nan, leftcop)
    # rightcop[rightcop < right_band] = np.nan
    # leftcop[leftcop < left_band] = np.nan

    # Optional manual point selection
    if manual_peaks is False:
        # Getting peaks and troughs
        xcom_peaks, _ = signal.find_peaks(xcom, width=width_threshold)
        xcom_troughs, _ = signal.find_peaks(-xcom, width=width_threshold)
    elif manual_peaks is True:
        xcom_peaks, _ = mos_marks(xcom, leftds, rightds, title="Select Peaks")
        xcom_troughs, _ = mos_marks(xcom, leftds, rightds, title="Select Troughs")
    else:
        print("The `manual` variable must be a boolean")

    lmos_values = np.array([])
    rmos_values = np.array([])

    lcop_points = leftcop[xcom_troughs]
    rcop_points = rightcop[xcom_peaks]

    for i in range(len(xcom_peaks)):
        # Getting window between peak values
        xcom_index = xcom_peaks[i]
        cop_point = rcop_points[i]

        # Getting non-nan values from region

        # Making sure we are actually grabbing the last meaningful region of center of pressure
        rmos = cop_point - xcom[xcom_index]
        # print(f"L COP {cop_point}")
        # print(f"xCoM {xcom[xcom_index]}")
        rmos_values = np.append(rmos_values, rmos)

    for i in range(len(xcom_troughs)):
        # Getting window between peak values
        xcom_index = xcom_troughs[i]
        cop_point = lcop_points[i]

        # Getting non-nan values from region
        lmos = xcom[xcom_index] - cop_point
        # print(f"R COP {cop_point}")
        # print(f"xCoM {xcom[xcom_index]}")
        lmos_values = np.append(lmos_values, lmos)

    return lmos_values, rmos_values, xcom_peaks, xcom_troughs


def limb_measurements(input_skeleton, skeleton_list, calibration):
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

    return calibrated_measurments


def main():

    # Loading in a dataset
    video = "00"
    df, bodyparts, scorer = dlck.load_data(
        f"../data/kinematics/EMG-test-1-pre-emg_0000{video}DLC_resnet50_dtr_update_predtxApr8shuffle1_1110000_filtered.h5"
    )

    # Loading in skeleton
    sk_df = pd.read_csv(
        f"../data/kinematics/EMG-test-1-pre-emg_0000{video}DLC_resnet50_dtr_update_predtxApr8shuffle1_1110000_filtered_skeleton.csv"
    )
    limb_names = [
        "iliac_crest_hip",
        "hip_knee",
        "knee_ankle",
        "ankle_metatarsal",
        "metatarsal_toe",
    ]

    # NOTE: Very important this is checked before running
    mouse_number = 2
    manual_analysis = False
    save_auto = False
    filter_k = 13

    # Settings before running initial workup from DeepLabCut
    figure_title = f"Step Cycles for level-test-M{mouse_number}-vid-{video}"
    figure_filename = f"./treadmill_level_test/emg-test-2-dropped/level_mos_analysis/m{mouse_number}-{video}.svg"
    step_cycles_filename = f"./treadmill_level_test/emg-test-2-dropped/level_mos_analysis/m{mouse_number}-step-cycles-{video}.csv"

    # Some things to set for plotting/saving
    lmos_filename = f"./treadmill_level_test/emg-test-2-dropped/level_mos_analysis/m{mouse_number}-lmos-{video}.csv"
    rmos_filename = f"./treadmill_level_test/emg-test-2-dropped/level_mos_analysis/m{mouse_number}-rmos-{video}.csv"
    mos_figure_title = (
        f"Measurement of Stability For Level Test M{mouse_number}-{video}"
    )
    mos_figure_filename = f"./treadmill_level_test/emg-test-2-dropped/level_mos_analysis/m{mouse_number}-mos-{video}.svg"
    calib_markers = [
        "calib_1",
        "calib_2",
        "calib_3",
        "calib_4",
        "calib_5",
        "calib_6",
    ]

    # For visualizing skeleton
    config_path = (
        "../../deeplabcut/dlc-dtr/dtr_update_predtx-kenzie-2024-04-08/config.yaml"
    )
    foi = "../data/kinematics/EMG-test-1-pre-emg_000000DLC_resnet50_dtr_update_predtxApr8shuffle1_1110000_filtered.h5"
    viz = Visualizer2D(config_path, foi, form_skeleton=True)
    viz.view(show_axes=True, show_grid=True, show_labels=True)
    # plt.show()

    calib_factor = dlc_calibrate(df, bodyparts, scorer, calib_markers)
    limb_diffs = limb_measurements(sk_df, limb_names, calib_factor)
    print(f"Length of limb coordinates in cm\n{limb_diffs}")

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
    time = frame_to_time(time)
    rfl_np = pd.array(rfl["y"])
    rfl_np = rfl_np / calib_factor
    rhl_np = pd.array(rhl["y"])
    rhl_np = rhl_np / calib_factor
    lfl_np = pd.array(lfl["y"])
    lfl_np = lfl_np / calib_factor
    lhl_np = pd.array(lhl["y"])
    lhl_np = lhl_np / calib_factor

    # Filtering to clean up traces like you would in spike
    toe_smooth = median_filter(toe_np, filter_k)
    toe_smooth = signal.savgol_filter(toe_smooth, 20, 3)
    # com_med = median_filter(comy_np, filter_k)
    com_med = signal.savgol_filter(comy_np, 40, 3)

    rfl_med = median_filter(rfl_np, filter_k)
    rfl_med = signal.savgol_filter(rfl_med, 30, 3)
    rhl_med = median_filter(rhl_np, filter_k)
    rhl_med = signal.savgol_filter(rhl_med, 30, 3)
    lfl_med = median_filter(lfl_np, filter_k)
    lfl_med = signal.savgol_filter(lfl_med, 30, 3)
    lhl_med = median_filter(lhl_np, filter_k)
    lhl_med = signal.savgol_filter(lhl_med, 30, 3)

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
    com_slope = spike_slope(com_trimmed, 30)
    hip_h = hip_height(toey_np, hipy_np)
    xcom_trimmed = xcom(com_trimmed, com_slope, hip_h)

    # Experimental Estimation of CoP considering the standards used
    rightcop = cop(rfl_med, rhl_med)
    rightcop = signal.savgol_filter(rightcop, 40, 3)
    leftcop = cop(lfl_med, lhl_med)
    leftcop = signal.savgol_filter(leftcop, 40, 3)
    right_DS = rightcop
    left_DS = leftcop

    # Calling function for swing estimation
    swing_onset, swing_offset = swing_estimation(toe_smooth)
    step_cyc_durations = step_cycle_est(toe_smooth)

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

    # plt.show()

    # Now onto Lateral stability

    lmos, rmos, xcom_peaks, xcom_troughs = mos(
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
    fig, axs = plt.subplots(2)
    fig.suptitle(mos_figure_title)

    # For plotting figure demonstrating how calculation was done
    axs[0].set_title("How MoS is Derived")
    axs[0].plot(xcom_trimmed)
    axs[0].plot(xcom_peaks, xcom_trimmed[xcom_peaks], "^")
    axs[0].plot(xcom_troughs, xcom_trimmed[xcom_troughs], "v")
    axs[0].plot(leftcop)
    axs[0].plot(rightcop)
    axs[0].legend(xcom_legend, bbox_to_anchor=(1, 0.7))

    # Looking at result
    axs[1].set_title("MoS Result")
    sns.barplot(data=mos_comb, x="Limb", y="MoS (cm)", ci=95, capsize=0.05, ax=axs[1])
    # axs[1].bar(0, np.mean(lmos), yerr=np.std(lmos), capsize=5)
    # axs[1].bar(1, np.mean(rmos), yerr=np.std(rmos), capsize=5)

    fig = plt.gcf()
    fig.set_size_inches(8.27, 11.7)  # A4 formatting 8.27” by 11.7”
    fig.tight_layout()

    # Saving results
    if manual_analysis is True:
        np.savetxt(lmos_filename, lmos, delimiter=",")
        np.savetxt(rmos_filename, rmos, delimiter=",")
        plt.savefig(mos_figure_filename, dpi=300)
        print("Mos results saved!")
    elif manual_analysis is False and save_auto is True:
        np.savetxt(lmos_filename, lmos, delimiter=",")
        np.savetxt(rmos_filename, rmos, delimiter=",")
        plt.savefig(mos_figure_filename, dpi=300)
        print("Mos results saved!")
    else:
        print("Mos results not saved")

    xcom

    # plt.show()


if __name__ == "__main__":
    main()
