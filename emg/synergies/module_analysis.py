# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from pandas import DataFrame as df
import seaborn as sns
# from scipy import stats as st
from statannotations.Annotator import Annotator
import motorpyrimitives as mp


def show_modules(data_input, chosen_synergies, modules_filename="./output.png"):
    """
    Make sure you check the channel order!!

    """

    # =======================================
    # Presenting Data as a mutliplot figure |
    # =======================================
    motor_primitives, motor_modules = synergy_extraction(data_input, chosen_synergies)
    channel_order = ['GM', 'Ip', 'BF', 'VL', 'St', 'TA', 'Gs', 'Gr']

    fig, axs = plt.subplots(chosen_synergies, 1, figsize=(4, 10))

    # Calculate the average trace for each column
    # samples = np.arange(0, len(motor_primitives))
    # samples_binned = np.arange(200)
    # number_cycles = len(motor_primitives) // 200

    for col in range(chosen_synergies):
        # primitive_trace = np.zeros(200)  # Initialize an array for accumulating the trace values

        # Begin Presenting Motor Modules

        # Get the data for the current column
        motor_module_column_data = motor_modules[col, :]  # Select all rows for the current column

        # Set the x-axis values for the bar graph
        x_values = np.arange(len(motor_module_column_data))

        # Plot the bar graph for the current column in the corresponding subplot
        axs[col].bar(x_values, motor_module_column_data)

        # Remove top and right spines of each subplot

        # Remove x and y axis labels and ticks from the avg_trace subplot
        axs[col].set_xticks([])
        axs[col].set_yticks([])
        axs[col].set_xlabel('')
        axs[col].set_ylabel('')
        axs[col].spines['top'].set_visible(False)
        axs[col].spines['right'].set_visible(False)

        # Remove x and y axis labels and ticks from the motor module subplot
        axs[col].set_xticks(x_values, channel_order)
        axs[col].set_yticks([])
        # axs[1, col].set_xlabel('')
        # axs[1, col].set_ylabel('')

    # Adjust spacing between subplots
    plt.tight_layout()
    # fig.suptitle(synergies_title, fontsize=16, fontweight='bold')
    # plt.savefig(modules_filename, dpi=300)
    # plt.subplots_adjust(top=0.9)
    plt.show()

# %%
# ================
# Synergy analysis
# ================

synergy_selection = 3

modules_wt = [
    './wt_data/wt_m1_non_modules.csv',
    './wt_data/wt_m1_per_modules.csv',
    './wt_data/wt_m2_non_modules.csv',
    './wt_data/wt_m2_per_modules.csv',
    './wt_data/wt_m3_non_modules.csv',
    './wt_data/wt_m3_per_modules.csv',
    './wt_data/wt_m4_non_modules.csv',
    './wt_data/wt_m4_per_modules.csv',
    './wt_data/wt_m5_non_modules.csv',
    './wt_data/wt_m5_per_modules.csv',
]

conditions_wt = [
    './norm-wt-m1-non.csv',
    './norm-wt-m1-per.csv',
    './norm-wt-m2-non.csv',
    './norm-wt-m2-per.csv',
    './norm-wt-m3-non.csv',
    './norm-wt-m3-per.csv',
    './norm-wt-m4-non.csv',
    './norm-wt-m4-per.csv',
    './norm-wt-m5-non.csv',
    './norm-wt-m5-per.csv',
]

modules_df = df(columns=["Condition", "Perturbation State", "Synergy", "GM", "Ip", "BF", "VL", "St", "TA", "Gs", "Gr"])

module_entry = []

for i in range(0, synergy_selection):
    current_synergy = i + 1
    synergy_tag = "Synergy {}".format(current_synergy)
    for j in range(0, len(conditions_wt)):
        condition_tag = "WT"
        if j == 0 or j % 2 == 0:
            perturbation_state_tag = "Non-Perturbation"
        else:
            perturbation_state_tag = "Perturbation"

        # motor_p, motor_m = mp.synergy_extraction(conditions_wt[j], synergy_selection)
        motor_data = pd.read_csv(modules_wt[j], header=None)
        motor_m_array = motor_data.to_numpy()
        current_module_set = motor_m_array[i, :]  # Select all rows for the current column
        identifiers = [condition_tag, perturbation_state_tag, synergy_tag]
        current_module_set = np.ndarray.tolist(current_module_set)
        entry = identifiers + current_module_set
        module_entry.append(entry)

columns = ["Condition", "Perturbation State", "Synergy", "GM", "Ip", "BF", "VL", "St", "TA", "Gs", "Gr"]
muscle_order = ["GM", "Ip", "BF", "VL", "St", "TA", "Gs", "Gr"]
condition_order = ["Non-Perturbation", "Perturbation"]
modules_df = pd.DataFrame.from_records(module_entry, columns=columns)
print(modules_df)

syn1_df = modules_df[modules_df["Synergy"] == "Synergy 1"]
syn2_df = modules_df[modules_df["Synergy"] == "Synergy 2"]
syn3_df = modules_df[modules_df["Synergy"] == "Synergy 3"]

modules_df_flip = modules_df.T
# modules_df.to_csv('./wt_modules.csv')
# modules_df_flip.to_csv('./wt_modules_flipped.csv')
# Plotting
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set(style="white", font_scale=1.5, rc=custom_params)

# modules_df.plot(kind="bar", hue=muscle_order)
# pairs = [
#     [("WT", "Non-Perturbation"), ("WT", "Perturbation")],
#     # [("PreDTX", "Non-Perturbation"), ("PreDTX", "Perturbation")],
#     # [("PostDTX", "Non-Perturbation"), ("PostDTX", "Perturbation")],
# ]
# 
# perturbation_state_order = ["Non-Perturbation", "Perturbation"]
# Plot for Synergy 1
print(syn1_df[["GM", "Ip", "BF", "VL", "St", "TA", "Gs", "Gr"]])
plot_params = {
    "data": syn1_df,
    "x": syn1_df[["GM", "Ip", "BF", "VL", "St", "TA", "Gs", "Gr"]],
    # "y": syn1_df["gm", "ip", "bf", "vl", "st", "ta", "gs", "gr"],
    "hue": "Perturbation State",
    "hue_order": condition_order,
}

plt.title("Modules For Synergy 1")
# plt.ylim(0, 1)
# plt.ylim(0, 250)
# syn1 = sns.barplot(data=syn1_df, capsize=0.05)
syn1 = sns.barplot(data=syn1_df, x="Condition", y="GM", hue="Perturbation State")
syn1 = sns.barplot(data=syn1_df, x="Condition", y="Ip", hue="Perturbation State")
syn1 = sns.barplot(data=syn1_df, x="Condition", y="BF", hue="Perturbation State")

# syn1 = sns.barplot(data=syn1_df, x="Condition", y="GM", hue="Perturbation State")
# syn1 = sns.barplot(data=syn1_df, x="Condition", y="GM", hue="Perturbation State")
# syn1 = sns.barplot(data=syn1_df, x="Condition", y="GM", hue="Perturbation State")
plt.ylabel('')
plt.legend(loc='best', fontsize=12)
# annotator = Annotator(syn1, pairs, **plot_params)
# annotator.new_plot(syn1, pairs, plot="barplot", **plot_params)
# annotator.configure(hide_non_significant=False, test="t-test_ind", text_format="star", loc="inside")
# annotator.apply_test().annotate(line_offset_to_group=0.2, line_offset=0.1)
# plt.show()






# Plot for Synergy 2
# plot_params = {
#     "data": syn2_df,
#     "x": "Condition",
#     "y": "Activation",
#     "hue": "Perturbation State",
#     "hue_order": perturbation_state_order,
# }
# plt.title("Activation For Synergy 2")
# plt.ylim(0, 250)
# syn2 = sns.barplot(**plot_params, ci="sd", capsize=0.05)
# plt.ylabel('')
# plt.legend(loc='best', fontsize=12)
# annotator = Annotator(syn2, pairs, **plot_params)
# annotator.new_plot(syn2, pairs, plot="barplot", **plot_params)
# annotator.configure(hide_non_significant=False, test="t-test_ind", text_format="star", loc="inside")
# annotator.apply_test().annotate(line_offset_to_group=0.2, line_offset=0.1)
# # plt.show()
# 
# # Plot for Synergy 3
# plot_params = {
#     "data": syn3_df,
#     "x": "Condition",
#     "y": "Activation",
#     "hue": "Perturbation State",
#     "hue_order": perturbation_state_order,
# }
# 
# plt.title("Activation For Synergy 3")
# plt.ylim(0, 250)
# syn3 = sns.barplot(**plot_params, ci="sd", capsize=0.05)
# plt.ylabel('')
# plt.legend(loc='best', fontsize=12)
# annotator = Annotator(syn3, pairs, **plot_params)
# annotator.new_plot(syn3, pairs, plot="barplot", **plot_params)
# annotator.configure(hide_non_significant=False, test="t-test_ind", text_format="star", loc="inside")
# annotator.apply_test().annotate(line_offset_to_group=0.2, line_offset=0.1)

# %%

