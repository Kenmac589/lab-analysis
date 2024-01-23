# This is another test

"""Average length of gait cycle per condition

This program is supposed to find the average

"""

import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
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

def extract_cycles(input_dataframe, swonset_channel="44 sw onset"):

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

def stance_duration(input_dataframe):

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



    return stance_periods

def hip_height(input_dataframe, toey="24 toey (cm)", hipy="16 Hipy (cm)"):

    # Bringing in the values for toey and hipy
    toey_values = input_dataframe[toey].tolist()
    hipy_values = input_dataframe[hipy].tolist()
    toey_values = np.array(toey_values)
    hipy_values = np.array(hipy_values)

    # Remove missing values
    toey_values = toey_values[np.logical_not(np.isnan(toey_values))]
    hipy_values = hipy_values[np.logical_not(np.isnan(hipy_values))]

    # Getting lower quartile value of toey as proxy for the ground
    toey_lowerq = np.percentile(toey_values, q=25)
    average_hip_value = np.mean(hipy_values)

    hip_height = average_hip_value - toey_lowerq
    return hip_height

def xcom(input_dataframe, hip_height, comy="37 CoMy (cm)"):

    # Bring in data
    comy_values = input_dataframe[comy].tolist()
    comy_values = np.array(comy_values)
    comy_values = comy_values[np.logical_not(np.isnan(comy_values))]

    # Getting slope of values
    vcom = np.diff(comy_values)

    # Fixing 1 off
    comy_values = np.delete(comy_values, - 1)

    # Get xCoM in (cm)
    xcom = comy_values + (vcom / np.sqrt(981 / hip_height))

    print("comy", comy_values.size)
    print("vcom", vcom.size)
    print("xcom", xcom)

    x_axis = np.arange(len(comy_values))


    # Testing output
    rbf = sp.interpolate.Rbf(x_axis, vcom, function='thin_plate', smooth=2)
    xnew = np.linspace(x_axis.min(), x_axis.max(), num=100, endpoint=True)
    ynew = rbf(xnew)

    fig, axs = plt.subplots(4, 1, layout='constrained')
    axs[0].set_title("CoMy")
    axs[0].legend(loc="best")
    axs[0].plot(x_axis, comy_values)
    axs[1].set_title("vCoM")
    axs[1].plot(x_axis, vcom)
    axs[2].set_title("Radial basis funtion interpolation of vCoM")
    axs[2].plot(xnew, ynew)
    axs[0].plot(x_axis, xcom)

    plt.show()

    return xcom


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
    cycle_results_csv = 'cycle_analysis.csv'

    with open(cycle_results_csv, 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the header row (optional)
        writer.writerow(['Data Point', 'Mean', 'Standard Deviation'])

        # Write data from the dictionary
        for key, (mean, std_dev) in cycle_results.items():
            writer.writerow([key, mean, std_dev])

    print(f'Data has been saved to {cycle_results_csv}')

# Main Code Body
def main():

    wt1nondf = pd.read_csv('./wt_1_non-perturbation.csv', header=0)
    wt_1_non_step_cycles = extract_cycles(wt1nondf)

    hipH = hip_height(wt1nondf, toey="24 toey", hipy="16 Hipy")

    wt1perdf = pd.read_csv('./wt_1_perturbation.csv', header=0)
    wt_1_per_step_cycles = extract_cycles(wt1perdf)

    hipH = hip_height(wt1perdf)
    xcomwtper = xcom(wt1perdf, hipH)

    wt4nondf = pd.read_csv('./wt_4_non-perturbation.csv')
    hipH = hip_height(wt4nondf)

if __name__ == "__main__":
    main()
