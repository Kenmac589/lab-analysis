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

        motor_p, motor_m = mp.synergy_extraction(conditions_wt[j], synergy_selection)
        # motor_data = pd.read_csv(modules_wt[j], header=None)
        # motor_m_array = motor_data.to_numpy()
        current_module_set = motor_m[i, :]  # Select all rows for the current column
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

# %%

# %%

