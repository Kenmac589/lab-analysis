"""Average length of gait cycle per condition

This program is supposed to find the average 

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import f_oneway

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
        time_diff = time_values[i] - time_values[i-1]
        time_differences.append(time_diff)

    # Finding the average value for the list
    time_differences_array = np.array(time_differences)
    treadmill_speed_array = np.array(treadmill_speed)

    # Creating masks to filter any values above 1 as this would be between distinct recordings
    recording_cutoff_high = 1.0
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

# Main Code Body

# Read in all csv's with cycle timing
directory_path = "./M5"
trial_list = read_all_csv(directory_path)

# Initialize Dictionary for storing results for each trial
cycle_results = {}
for key in trial_list:
    cycle_results[key] = None

# Now, you can access the data from each file like this:
for filename, data in trial_list.items():
    step_duration_array, treadmill_speed = step_duration(data)
    cycle_results[filename] = np.mean(step_duration_array), np.mean(treadmill_speed)
    print(cycle_results[filename])
    print(f"Average step durations for {filename}:", np.mean(step_duration_array))
    print(f"Treadmill speed for {filename}:", np.mean(treadmill_speed))
    # print(f"Data from {filename}:")
    # print(data.head)



