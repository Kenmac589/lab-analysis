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

# Main Code Body
M3_non_low = pd.read_csv("./M3-non-100.csv", header=0)
M3_non_high = pd.read_csv("./M3-non-200.csv", header=0)
M3_per_low = pd.read_csv("./M3-per-100.csv", header=0)
M3_per_high = pd.read_csv("./M3-per-200.csv", header=0)

# sinusoidal = pd.read_csv("./sinusoidal.csv", header=0)

non_perturbation_step_duration_low, non_perturbation_treadmill_low = step_duration(non_perturbation_low)
non_perturbation_step_duration_high, non_perturbation_treadmill_high = step_duration(non_perturbation_high)
perturbation_step_duration_low, perturbation_treadmill_low = step_duration(perturbation_low)
perturbation_step_duration_high, perturbation_treadmill_high = step_duration(perturbation_high)
# sinusoidal_step_duration, sinusoidal_treadmill = step_duration(sinusoidal)

print("Non-perturbation at speed 0.100 m/sec:", np.mean(non_perturbation_step_duration_low))
print("Non-perturbation at speed 0.200 m/sec:",  np.mean(non_perturbation_step_duration_high))
print("Perturbation at speed 0.100 m/sec:",  np.mean(perturbation_step_duration_low))
print("Perturbation at speed 0.200 m/sec:",  np.mean(perturbation_step_duration_high))

# print(np.mean(sinusoidal_step_duration))

# Breaking up into individual recordings based on speed
# speed_ranges = [0.000, 0.225, 0.275, 0.325, 0.375]
# 
# bin_indicies = np.digitize(non_perturbation_treadmill, speed_ranges) - 1
# 
# # Compute the number of bins
# num_bins = len(speed_ranges) - 1
# 
# # Sort the indicies based on bin indicies
# sorted_indicies = np.argsort(bin_indicies)
# 
# binned_treadmill = np.bincount(bin_indicies, minlength=num_bins)
# binned_non_pertubation = np.bincount(bin_indicies, weights=non_perturbation_step_duration, minlength=num_bins)
# 
# 
# # Reshape the binned data into 2D array
# reshaped_treadmill = binned_treadmill.reshape(1 , -1)
# reshaped_step_difference = binned_non_pertubation.reshape(1, -1)
# 
# print(reshaped_treadmill)
# print(reshaped_step_difference)

# Plotting the results
# Finding where the cutoff values occur in the differential array
# for i in range(len(time_differences_array)):  
#     if time_differences_array[i] >= 1:
#         print(i)
print(non_perturbation_step_duration_low)


# Combine the data into a list
data = [non_perturbation_step_duration_low, perturbation_step_duration_low, non_perturbation_step_duration_high, perturbation_step_duration_high]

# Create x-coordinates for the box plots
x = np.arange(1, len(data) + 1)

# Plot the box and whisker plots
plt.boxplot(data, positions=x)

# Add x-axis labels
plt.xticks(x, ['non-per-100', 'per-100', 'non-per-200', 'per-200'])

# Add a title
plt.title('Multiple Box and Whisker Plots')

# Display the plot
plt.show()


# Create a figure and axes for subplots

fig, ax = plt.subplots(1, 4, figsize=(10, 4))

# Plot the box and whisker plots
ax[0].boxplot(non_perturbation_step_duration_low)
ax[0].set_title('non-per-100')

ax[1].boxplot(perturbation_step_duration_low)
ax[1].set_title('per-100')

ax[2].boxplot(non_perturbation_step_duration_high)
ax[2].set_title('non-per-200')

ax[3].boxplot(perturbation_step_duration_high)
ax[3].set_title('per-200')

# Adjust the spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()

# Plot the time differences
# plt.boxplot(non_perturbation_step_duration_low)
# plt.boxplot(perturbation_step_duration_low)
# plt.boxplot(non_perturbation_step_duration_high)
# plt.boxplot(perturbation_step_duration_high)
# plt.title('Step Cycle')
# plt.show()
