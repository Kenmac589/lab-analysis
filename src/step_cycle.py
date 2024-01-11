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

def cycle_periods(input_dataframe):

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

    return adjusted_time_differences

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

    # subject = (
    #     "WT-M1",

    # )
    data_dict = {}  # Initialize an empty dictionary to store the data
    data_dict['WT-M1 Non-Perturbation'] = pd.read_csv('./CoM-M1/WT-M1 without Perturbation.txt')
    data_dict['WT-M1 Perturbation'] = pd.read_csv('./CoM-M1/WT-M1 with Perturbation.txt')
    data_dict['WT-M2 Non-Perturbation'] = pd.read_csv('./CoM-M2/CoM-M2 without Perturbation.txt')
    data_dict['WT-M2 Perturbation'] = pd.read_csv('./CoM-M2/CoM-M2 with Perturbation.txt')
    data_dict['WT-M3 Non-Perturbation'] = pd.read_csv('./CoM-M3/WT-M3 without Peturbation.txt')
    data_dict['WT-M3 Perturbation'] = pd.read_csv('./CoM-M3/WT-M3 with Perturbation.txt')
    # data_dict['PreDTX Non-Perturbation'] = pd.read_csv('./M5/PreDTX Without Perturbation.csv')
    # data_dict['PreDTX Perturbation'] = pd.read_csv('./M5/PreDTX With Perturbation.csv')
    # data_dict['PostDTX Non-Perturbation'] = pd.read_csv('./M5/PostDTX Without Perturbation.csv')
    # data_dict['PostDTX Perturbation'] = pd.read_csv('./M5/PostDTX With Perturbation.csv')

    # Read in all csv's with cycle timing
    # This is all that really has to change
    # directory_path = "./M5"
    trial_list = data_dict

    # cycle_period_summary(directory_path)
    cycle_results_df = pd.DataFrame()

    # Initialize Dictionary for storing results for each trial
    cycle_results = {}
    for key in trial_list:
        cycle_results[key] = None
        cycle_results_df[key] = None

    # Keeping the keys as a list of strings for iteration purposes
    trials = list(cycle_results.keys())

    # Now, you can access the data from each file like this:
    for filename, data in trial_list.items():
        step_duration_array = cycle_periods(data)
        cycle_results[filename] = step_duration_array

    # Convert Dictionary of Results to Dataframe
    cycle_results_df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in cycle_results.items()]))

    cycle_results_df.to_csv("./cycle_comparisons.csv")
    # pairs = [
    #     ('WT Non-Perturbation', 'WT Perturbation'),
    #     ('PreDTX Non-Perturbation', 'WT Non-Perturbation'),
    #     ('WT Non-Perturbation', 'PostDTX Non-Perturbation'),
    #     ('PreDTX Perturbation', 'WT Perturbation'),
    #     ('PreDTX Non-Perturbation', 'PreDTX Perturbation'),
    #     ('PreDTX Non-Perturbation', 'PostDTX Non-Perturbation'),
    #     ('PreDTX Non-Perturbation', 'PostDTX Perturbation'),
    #     ('PreDTX Perturbation', 'PostDTX Non-Perturbation'),
    #     ('PreDTX Perturbation', 'PostDTX Perturbation'),
    #     ('PostDTX Non-Perturbation', 'PostDTX Perturbation'),
    # ]

    non_per = cycle_results_df.loc[:, [col for col in cycle_results_df.columns if 'Non-Perturbation' in col]]
    # per = cycle_results_df.loc[:, [col for col in cycle_results_df.columns if 'Perturbation' in col]]
    # Plotting
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set(style="white", rc=custom_params)

    plt.title("Step Cycle Durations WT vs DTR M5")
    plt_cyc = sns.barplot(
        x=non_per.columns,
        y=non_per.mean(),
        order=non_per.columns,
        zorder=2
    )
    plt_cyc.errorbar(x=non_per.columns, y=non_per.mean(), yerr=non_per.std(), capsize=3, fmt="none", c="k", zorder=1)
    # annotator = Annotator(plt_cyc, pairs, data=cycle_results_df)
    # annotator.configure(hide_non_significant=True, test='t-test_welch', text_format='simple')
    # annotator.apply_test().annotate(line_offset_to_group=0.2, line_offset=0.1)
    # annotator.apply_and_annotate()
    plt.show()

if __name__ == "__main__":
    main()
