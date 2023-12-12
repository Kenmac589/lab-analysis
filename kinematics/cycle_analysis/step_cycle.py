"""Average length of gait cycle per condition

This program is supposed to find the average 

"""

import os
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

    return adjusted_time_differences

def conditional_cycle(file_paths, cycle_df, condition_tag):
    """
    @param input_data: Pandas Dataframe

    Make sure you load the input files where it is all one state first then all the next
    """
    # trials_per_state = len(file_paths) / 2
    perturbation_state = ["Non-Perturbation", "Perturbation"]
    # condition_tag = str(condition)

    # Going through first half for Non-Perturbation
    for i in range(len(file_paths) // 2):
        trial_df = pd.read_csv(file_paths[i], header=0)
        pertubation_state_tag = perturbation_state[0]
        step_cycles = step_duration(trial_df)
        for j in range(len(step_cycles)):
            cycle_entry = [[condition_tag, pertubation_state_tag, step_cycles[j]]]
            cycle_df = cycle_df.append(pd.DataFrame(cycle_entry, columns=["Condition", "Perturbation State", "Step Cycle Duration"]), ignore_index=True)

    # Going through second half for Perturbation
    for i in range(len(file_paths) // 2, len(file_paths)):
        trial_df = pd.read_csv(conditions[i], header=0)
        pertubation_state_tag = perturbation_state[1]
        step_cycles = step_duration(trial_df)
        for j in range(len(step_cycles)):
            cycle_entry = [[condition_tag, pertubation_state_tag, step_cycles[j]]]
            cycle_df = cycle_df.append(pd.DataFrame(cycle_entry, columns=["Condition", "Perturbation State", "Step Cycle Duration"]), ignore_index=True)

    return cycle_df
# Main Code Body
# conditions = ["WT", "PreDTX", "PostDTX"]

# Dataset assignments
conditions = {
    "WT": [
        "../../kinematics/cycle_analysis/CoM-M1/WT-M1 without Perturbation.txt",
        "../../kinematics/cycle_analysis/CoM-M2/CoM-M2 without Perturbation.txt",
        "../../kinematics/cycle_analysis/CoM-M3/WT-M3 without Peturbation.txt",
        "../../kinematics/cycle_analysis/CoM-M1/WT-M1 with Perturbation.txt",
        "../../kinematics/cycle_analysis/CoM-M2/CoM-M2 with Perturbation.txt",
        "../../kinematics/cycle_analysis/CoM-M3/WT-M3 with Perturbation.txt",
    ],
    "PreDTX": [
        "../../kinematics/cycle_analysis/DTR-M5/PreDTX Without Perturbation.csv",
        "../../kinematics/cycle_analysis/DTR-M5/PreDTX with Perturbation.csv",
    ],
    "PostDTX": [
        "../../kinematics/cycle_analysis/DTR-M5/PostDTX without Perturbation.csv",
        "../../kinematics/cycle_analysis/DTR-M5/PostDTX with Perturbation.csv",
    ]
}

conditions_names_order = ["WT", "PreDTX", "PostDTX"]
# trials = list(conditions.keys())
# test_df = pd.DataFrame(columns=["Condition", "Perturbation State", "Step Cycle Duration"])
# 
# for i in conditions:
#     file_list = conditions[i]
#     test_df = conditional_cycle(file_list, test_df, "WT")
# 
# print(test_df)

conditions = [
    "../../kinematics/cycle_analysis/CoM-M1/WT-M1 without Perturbation.txt",
    "../../kinematics/cycle_analysis/CoM-M2/CoM-M2 without Perturbation.txt",
    "../../kinematics/cycle_analysis/CoM-M3/WT-M3 without Peturbation.txt",
    "../../kinematics/cycle_analysis/DTR-M5/PreDTX Without Perturbation.csv",
    "../../kinematics/cycle_analysis/DTR-M5/PostDTX without Perturbation.csv",
    "../../kinematics/cycle_analysis/CoM-M1/WT-M1 with Perturbation.txt",
    "../../kinematics/cycle_analysis/CoM-M2/CoM-M2 with Perturbation.txt",
    "../../kinematics/cycle_analysis/CoM-M3/WT-M3 with Perturbation.txt",
    "../../kinematics/cycle_analysis/DTR-M5/PreDTX with Perturbation.csv",
    "../../kinematics/cycle_analysis/DTR-M5/PostDTX with Perturbation.csv",
]

# Giving them nice tags
conditions_names = ["WT", "WT", "WT", "PreDTX", "PostDTX", "WT", "WT", "WT", "PreDTX", "PostDTX"]
perturbation_state = ["Non-Perturbation", "Non-Perturbation", "Non-Perturbation", "Non-Perturbation", "Non-Perturbation", "Perturbation", "Perturbation", "Perturbation", "Perturbation", "Perturbation"]

conditions_names_order = ["WT", "PreDTX", "PostDTX"]
perturbation_state_order = ["Non-Perturbation", "Perturbation"]


cycle_means = []

cycle_df = pd.DataFrame(columns=["Condition", "Perturbation State", "Step Cycle Duration"])

for i in range(len(conditions)):
    condition_tag = conditions_names[i]
    pertubation_state_tag = perturbation_state[i]
    trial_df = pd.read_csv(conditions[i], header=0)
    step_cycles = step_duration(trial_df)

    for j in range(len(step_cycles)):
        cycle_entry = [[condition_tag, pertubation_state_tag, step_cycles[j], np.std(step_cycles)]]
        cycle_df = cycle_df._append(pd.DataFrame(cycle_entry, columns=["Condition", "Perturbation State", "Step Cycle Duration", "Error"]), ignore_index=True)


# Plotting
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set(style="white", font_scale=1.5, rc=custom_params)

pairs = [
    [("WT", "Non-Perturbation"), ("WT", "Perturbation")],
    [("PreDTX", "Non-Perturbation"), ("PreDTX", "Perturbation")],
    [("PostDTX", "Non-Perturbation"), ("PostDTX", "Perturbation")],
]

plot_params = {
    "data": cycle_df,
    "x": "Condition",
    "y": "Step Cycle Duration",
    "hue": "Perturbation State",
    "hue_order": perturbation_state_order,
}

plt.title("Step Cycle Duration")
plt.ylim(0, 1)
cycle = sns.barplot(**plot_params, ci="sd", capsize=0.05)
plt.ylabel('')
plt.legend(loc='best', fontsize=12)
annotator = Annotator(cycle, pairs, **plot_params)
annotator.new_plot(cycle, pairs, plot="barplot", **plot_params)
annotator.configure(hide_non_significant=False, test="t-test_ind", text_format="star", loc="inside")
annotator.apply_test().annotate(line_offset_to_group=0.2, line_offset=0.1)
plt.show()

