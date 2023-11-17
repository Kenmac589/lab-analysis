"""Average length of gait cycle per condition

This program is supposed to find the average 

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

kinematicData = pd.read_csv("./com-m3-wt-20220420-swing_and_stance.csv", header=0)
kinData = np.array(kinematicData)

# Define the value and column to search for
value_to_find = 1
column_to_search = "44 sw onset"
column_for_time = "Time"

# Store time values when the specified value is found
time_values = []

# Iterate through the DataFrame and process matches
for index, row in kinematicData.iterrows():
    if row[column_to_search] == value_to_find:
        time_value = row[column_for_time]
        time_values.append(time_value)

# Calculate the differences between consecutive time values
time_differences = []
for i in range(len(time_values)):
    time_diff = time_values[i] - time_values[i-1]
    time_differences.append(time_diff)

# print(len(time_values))
# print(len(time_differences))
# Finding the average value for the list
time_differences_array = np.array(time_differences)

# Creating mask to filter any values above 1 as this would be between distinct recordings
recording_cutoff = 1.0

# Applying the filter to the array
adjusted_time_differences = time_differences_array <= recording_cutoff
average_step_difference = np.mean(adjusted_time_differences)

print(average_step_difference)
print(adjusted_time_differences)

# Some exploration!

# Finding where the cutoff values occur in the differential array
# for i in range(len(time_differences_array)):
#     if time_differences_array[i] >= 1:
#         print(i)

# Plot the time differences
# plt.plot(time_differences_array)
# plt.xlabel('Index')
# plt.ylabel('Time Difference')
# plt.title('Time Differences Plot')
# plt.show()
