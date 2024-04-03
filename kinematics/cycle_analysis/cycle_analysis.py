import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import latstability as ls

data_dict = {}  # Initialize an empty dictionary to store the data
data_dict["WT-M1 Non-Perturbation"] = pd.read_csv(
    "./CoM-M1/WT-M1 without Perturbation.txt"
)
data_dict["WT-M1 Perturbation"] = pd.read_csv("./CoM-M1/WT-M1 with Perturbation.txt")
data_dict["WT-M2 Non-Perturbation"] = pd.read_csv(
    "./CoM-M2/CoM-M2 without Perturbation.txt"
)
data_dict["WT-M2 Perturbation"] = pd.read_csv("./CoM-M2/CoM-M2 with Perturbation.txt")
data_dict["WT-M3 Non-Perturbation"] = pd.read_csv(
    "./CoM-M3/WT-M3 without Peturbation.txt"
)
data_dict["WT-M3 Perturbation"] = pd.read_csv("./CoM-M3/WT-M3 with Perturbation.txt")
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
    step_duration_array = ls.extract_cycles(data)
    cycle_results[filename] = step_duration_array

# Convert Dictionary of Results to Dataframe
cycle_results_df = pd.DataFrame(
    dict([(key, pd.Series(value)) for key, value in cycle_results.items()])
)

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

non_per = cycle_results_df.loc[
    :, [col for col in cycle_results_df.columns if "Non-Perturbation" in col]
]
# per = cycle_results_df.loc[:, [col for col in cycle_results_df.columns if 'Perturbation' in col]]
# Plotting
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set(style="white", rc=custom_params)

plt.title("Step Cycle Durations WT vs DTR M5")
plt_cyc = sns.barplot(
    x=non_per.columns, y=non_per.mean(), order=non_per.columns, zorder=2
)
plt_cyc.errorbar(
    x=non_per.columns,
    y=non_per.mean(),
    yerr=non_per.std(),
    capsize=3,
    fmt="none",
    c="k",
    zorder=1,
)
# annotator = Annotator(plt_cyc, pairs, data=cycle_results_df)
# annotator.configure(hide_non_significant=True, test='t-test_welch', text_format='simple')
# annotator.apply_test().annotate(line_offset_to_group=0.2, line_offset=0.1)
# annotator.apply_and_annotate()
plt.show()
