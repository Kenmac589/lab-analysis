"""Average length of gait cycle per condition

This program is supposed to find the average 

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def step_duration(input_dataframe):
    """
    @param: pandas dataframe

    """

    # Define the value and column to search for
    value_to_find = 1
    column_to_search = "44 sw onset"
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

non_perturbation = pd.read_csv("./non-perturbation.csv", header=0)
perturbation = pd.read_csv("./perturbation.csv")
sinusoidal = pd.read_csv("./sinusoidal.csv", header=0)

non_perturbation_step_duration, non_perturbation_treadmill = step_duration(non_perturbation)
perturbation_step_duration, perturbation_treadmill = step_duration(perturbation)
sinusoidal_step_duration, sinusoidal_treadmill = step_duration(sinusoidal)

print(np.mean(non_perturbation_step_duration))
print(np.mean(perturbation_step_duration))
print(np.mean(sinusoidal_step_duration))

# Breaking up into individual recordings


# Some exploration!

# Finding where the cutoff values occur in the differential array
# for i in range(len(time_differences_array)):
#     if time_differences_array[i] >= 1:
#         print(i)

# Plot the time differences
# plt.scatter(adjusted_treadmill_speeds, adjusted_time_differences)
# plt.xlabel('Treadmill Speeds')
# plt.ylabel('Time Difference')
# plt.title('Time Differences Plot')
# plt.show()
