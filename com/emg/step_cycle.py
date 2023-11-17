"""Average length of gait cycle per condition

This program is supposed to find the average 

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import f_oneway

def step_duration(input_dataframe):

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

# Read in M1 data
M1_non_perturbation_low = pd.read_csv("./M1-non-100.csv", header=0)
M1_non_perturbation_high = pd.read_csv("./M1-non-300.csv", header=0)
M1_perturbation_low = pd.read_csv("./M1-per-100.csv", header=0)
M1_perturbation_high = pd.read_csv("./M1-per-300.csv", header=0)

# Read in M2 data
M2_non_perturbation_low = pd.read_csv("./M2-non-100.csv", header=0)
M2_non_perturbation_high = pd.read_csv("./M2-non-200.csv", header=0)
M2_perturbation_low = pd.read_csv("./M2-per-100.csv", header=0)

# Read in M3 data
M3_non_perturbation_low = pd.read_csv("./M3-non-100.csv", header=0)
M3_non_perturbation_high = pd.read_csv("./M3-non-200.csv", header=0)
M3_perturbation_low = pd.read_csv("./M3-per-100.csv", header=0)
M3_perturbation_high = pd.read_csv("./M3-per-200.csv", header=0)

# Apply the step_duration function
M1_non_low_step_duration, M1_non_low_treadmill = step_duration(M1_non_perturbation_low)
M1_non_high_step_duration, M1_non_high_treadmill = step_duration(M1_non_perturbation_high)
M1_per_low_step_duration, M1_per_low_treadmill = step_duration(M1_perturbation_low)
M1_per_high_step_duration, M1_per_high_treadmill = step_duration(M1_perturbation_high)

M2_non_low_step_duration, M2_non_low_treadmill = step_duration(M2_non_perturbation_low)
M2_non_high_step_duration, M2_non_high_treadmill = step_duration(M2_non_perturbation_high)
M2_per_low_step_duration, M2_per_low_treadmill = step_duration(M2_perturbation_low)

M3_non_low_step_duration, M3_non_low_treadmill = step_duration(M3_non_perturbation_low)
M3_non_high_step_duration, M3_non_high_treadmill = step_duration(M3_non_perturbation_high)
M3_per_low_step_duration, M3_per_low_treadmill = step_duration(M3_perturbation_low)
M3_per_high_step_duration, M3_per_high_treadmill = step_duration(M3_perturbation_high)

results = {}

# Presenting average step duration
print("Non-Perturbation step duration for M1 at speed 0.100 m/sec:", np.mean(M1_non_low_step_duration), "standard deviation:", np.std(M1_non_low_step_duration))
print("Non-Perturbation step duration for M1 at speed 0.300 m/sec:", np.mean(M1_non_high_step_duration), "standard deviation:", np.std(M1_non_high_step_duration))
print("Pertubation step duration for M1 at speed 0.100 m/sec:", np.mean(M1_per_low_step_duration), "standard deviation:", np.std(M1_per_low_step_duration))
print("Perturbation duration for M1 at speed 0.300 m/sec:", np.mean(M1_per_high_step_duration), "standard deviation:", np.std(M1_per_high_step_duration))
print()
print("Non-Perturbation step duration for M2 at speed 0.100 m/sec:", np.mean(M2_non_low_step_duration), "standard deviation:", np.std(M2_non_low_step_duration))
print("Non-Perturbation step duration for M2 at speed 0.200 m/sec:", np.mean(M2_non_high_step_duration), "standard deviation:", np.std(M2_non_high_step_duration))
print("Perturbation step duration for M2 at speed 0.100 m/sec:", np.mean(M2_per_low_step_duration), "standard deviation:", np.std(M2_per_low_step_duration))
print()
print("Non-perturbation for M3 at speed 0.100 m/sec:", np.mean(M3_non_low_step_duration), "standard deviation:", np.std(M3_non_low_step_duration))
print("Non-perturbation for M3 at speed 0.200 m/sec:",  np.mean(M3_non_high_step_duration), "standard deviation:", np.std(M3_non_high_step_duration))
print("Perturbation for M3 at speed 0.100 m/sec:",  np.mean(M3_per_low_step_duration), "standard deviation:", np.std(M3_per_low_step_duration))
print("Perturbation for M3 at speed 0.200 m/sec:",  np.mean(M3_per_high_step_duration), "standard deviation:", np.std(M3_per_high_step_duration))

# Performing stats for each set of speeds

# T-test for effect of perturbation
M1_low_ttest = ttest_ind(M1_non_low_step_duration, M1_per_low_step_duration)
M1_high_ttest = ttest_ind(M1_non_high_step_duration, M1_per_high_step_duration)
M2_low_ttest = ttest_ind(M2_non_low_step_duration, M2_per_low_step_duration)
M3_low_ttest = ttest_ind(M3_non_low_step_duration, M3_per_low_step_duration)
M3_high_ttest = ttest_ind(M3_non_high_step_duration, M3_per_high_step_duration)

print("M1 test for 100 m/sec", M1_low_ttest)
print("M1 test for 300 m/sec", M1_high_ttest)
print("M2 test for 100 m/sec", M2_low_ttest)
print("M3 test for 100 m/sec", M3_low_ttest)
print("M3 test for 200 m/sec", M3_high_ttest)

# Combine the data into a list
M1_data = [M1_non_low_step_duration, M1_per_low_step_duration, M1_non_high_step_duration, M1_per_high_step_duration]
M2_data = [M2_non_low_step_duration, M2_per_low_step_duration, M2_non_high_step_duration]
M3_data = [M3_non_low_step_duration, M3_per_low_step_duration, M3_non_high_step_duration, M3_per_high_step_duration]

# Combine the data into a list
cumulative_data = [M1_non_low_step_duration, M1_per_low_step_duration, M1_non_high_step_duration, M1_per_high_step_duration, M2_non_low_step_duration, M2_per_low_step_duration, M2_non_high_step_duration, M3_non_low_step_duration, M3_per_low_step_duration, M3_non_high_step_duration, M3_per_high_step_duration]

means = [np.mean(i) for i in cumulative_data]
stds = [np.std(i) for i in cumulative_data]

# Create a figure and axes for subplots
x = np.arange(len(cumulative_data))

# Add x-axis labels
plt.xticks(x, ['M1-non-100', 'M1-per-100', 'M1-non-300', 'M1-per-300', 'M2-non-100', 'M2-per-100', 'M2-per-100', 'M3-non-100', 'M3-non-300', 'M3-per-100', 'M3-per-300'])

# Add a title
plt.title('Average step cycle by treadmill speed')

# Creating a second boxplot with error bars
means = [np.mean(i) for i in cumulative_data]
stds = [np.std(i) for i in cumulative_data]

x = np.arange(len(cumulative_data))

# Plot the bar graph
plt.bar(x, means)

# Plot the error bars
plt.errorbar(x, means, yerr=stds, fmt='.k', capsize=3)

# Display the plot
plt.show()

