import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame as df
import seaborn as sns
# from scipy import stats as st
from statannotations.Annotator import Annotator
import motorpyrimitives as mp

# ================
# Synergy analysis
# ================

synergy_selection = 3


conditions_wt_non = [
    './norm-wt-m1-non.csv',
    './norm-wt-m2-non.csv',
    './norm-wt-m3-non.csv',
    './norm-wt-m4-non.csv',
    './norm-wt-m5-non.csv',
]

conditions_wt_per = [
    './norm-wt-m1-per.csv',
    './norm-wt-m2-per.csv',
    './norm-wt-m3-per.csv',
    './norm-wt-m4-per.csv',
    './norm-wt-m5-per.csv',
]


# Title Names
title_names = [
    "Muscle Synergies for DTR-M5 preDTX without perturbation",
    "Muscle Synergies for DTR-M5 preDTX without perturbation",
    "Muscle Synergies for DTR-M5 preDTX without perturbation",
    "Muscle Synergies for DTR-M5 preDTX without perturbation",

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

# Normalized Data List
conditions_normalized_dtr = [
    './norm-emg-preDTX-100.csv',
    './norm-emg-preDTX-per.csv',
    './norm-postdtx-non.csv',
    './norm-postdtx-per.csv',
]

# Cleaned up Primitives
conditions_primitives_dtr = [
    './predtx-non-primitives-test.txt',
    './predtx-per-primitives-test.txt',
    './postdtx-non-primitives.txt',
    './postdtx-per-primitives.txt',
]

# ================================
# Full Width Half Maximum Analysis
# ================================

conditions_wt_primitives = [
    './wt-m1-non-primitives.txt',
    './wt-m1-per-primitives.txt',
    './wt-m2-non-primitives.txt',
    './wt-m2-per-primitives.txt',
    './wt-m3-non-primitives.txt',
    './wt-m3-per-primitives.txt',
    './wt-m4-non-primitives.txt',
    './wt-m4-per-primitives.txt',
    './wt-m5-non-primitives.txt',
    './wt-m5-per-primitives.txt',
]



modules_df = df(columns=['Condition', 'Perturbation State', 'Synergy', 'GM', 'Ip', 'BF', 'VL', 'St', 'TA', 'Gs', 'Gr'])
fwhm_df = df(columns=["Condition", "Perturbation State", "Synergy", "FWHM"])

# Loading all WT's first

for i in range(0, synergy_selection):
    current_synergy = i + 1
    synergy_tag = "Synergy {}".format(current_synergy)
    for j in range(0, len(conditions_wt_primitives)):
        condition_tag = "WT"
        if j == 0 or j % 2 == 0:
            perturbation_state_tag = "Non-Perturbation"
        else:
            perturbation_state_tag = "Perturbation"

        motor_p_data = pd.read_csv(conditions_wt_primitives[j], header=0)
        motor_p_array = motor_p_data.to_numpy()
        # motor_p_array, motor_m = mp.synergy_extraction(conditions_wt[j], current_synergy)
        fwhm_list = mp.fwhm(motor_p_array, current_synergy)
        for k in range(0, len(fwhm_list)):
            fwhm_entry = [[condition_tag, perturbation_state_tag, synergy_tag, fwhm_list[k]]]
            fwhm_df = fwhm_df._append(pd.DataFrame(fwhm_entry, columns=["Condition", "Perturbation State", "Synergy", "FWHM"]), ignore_index=True)

# For DTR
conditions = [
    './predtx-non-primitives-test.txt',
    './predtx-per-primitives-test.txt',
    './postdtx-non-primitives.txt',
    './postdtx-per-primitives.txt',
]

conditions_name = [
    "PreDTX",
    "PreDTX",
    "PostDTX",
    "PostDTX"
]

for i in range(0, synergy_selection):
    current_synergy = i + 1
    synergy_tag = "Synergy {}".format(current_synergy)
    for j in range(0, len(conditions)):
        condition_tag = conditions_name[j]
        if j == 0 or j % 2 == 0:
            perturbation_state_tag = "Non-Perturbation"
        else:
            perturbation_state_tag = "Perturbation"
        # print(condition_tag + " " + perturbation_state_tag)
        motor_p_data = pd.read_csv(conditions[j], header=0)
        motor_p_array = motor_p_data.to_numpy()

        # motor_p_array, motor_m = mp.synergy_extraction(conditions_wt[j], current_synergy)
        fwhm_list = mp.fwhm(motor_p_array, current_synergy)
        for k in range(0, len(fwhm_list)):
            fwhm_entry = [[condition_tag, perturbation_state_tag, synergy_tag, fwhm_list[k]]]
            fwhm_df = fwhm_df._append(pd.DataFrame(fwhm_entry, columns=["Condition", "Perturbation State", "Synergy", "FWHM"]), ignore_index=True)


# fwhm_df.to_csv("./fwhm_output_text.csv")
syn1_df = fwhm_df[fwhm_df["Synergy"] == "Synergy 1"]
syn2_df = fwhm_df[fwhm_df["Synergy"] == "Synergy 2"]
syn3_df = fwhm_df[fwhm_df["Synergy"] == "Synergy 3"]

# Plotting
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set(style="white", font_scale=1.5, rc=custom_params)

pairs = [
    [("WT", "Non-Perturbation"), ("WT", "Perturbation")],
    [("PreDTX", "Non-Perturbation"), ("PreDTX", "Perturbation")],
    [("PostDTX", "Non-Perturbation"), ("PostDTX", "Perturbation")],
]

perturbation_state_order = ["Non-Perturbation", "Perturbation"]

# Plot for Synergy 1
plot_params = {
    "data": syn1_df,
    "x": "Condition",
    "y": "FWHM",
    "hue": "Perturbation State",
    "hue_order": perturbation_state_order,
}

plt.title("FWHM For Synergy 1")
plt.ylim(0, 1)
plt.ylim(0, 250)
syn1 = sns.barplot(**plot_params, ci="sd", capsize=0.05)
plt.ylabel('')
plt.legend(loc='best', fontsize=12)
annotator = Annotator(syn1, pairs, **plot_params)
annotator.new_plot(syn1, pairs, plot="barplot", **plot_params)
annotator.configure(hide_non_significant=False, test="t-test_ind", text_format="star", loc="inside")
annotator.apply_test().annotate(line_offset_to_group=0.2, line_offset=0.1)
plt.show()


# Plot for Synergy 2
plot_params = {
    "data": syn2_df,
    "x": "Condition",
    "y": "FWHM",
    "hue": "Perturbation State",
    "hue_order": perturbation_state_order,
}
plt.title("FWHM For Synergy 2")
plt.ylim(0, 250)
syn2 = sns.barplot(**plot_params, ci="sd", capsize=0.05)
plt.ylabel('')
plt.legend(loc='best', fontsize=12)
annotator = Annotator(syn2, pairs, **plot_params)
annotator.new_plot(syn2, pairs, plot="barplot", **plot_params)
annotator.configure(hide_non_significant=False, test="t-test_ind", text_format="star", loc="inside")
annotator.apply_test().annotate(line_offset_to_group=0.2, line_offset=0.1)
plt.show()

# Plot for Synergy 3
plot_params = {
    "data": syn3_df,
    "x": "Condition",
    "y": "FWHM",
    "hue": "Perturbation State",
    "hue_order": perturbation_state_order,
}

plt.title("FWHM For Synergy 3")
plt.ylim(0, 250)
syn3 = sns.barplot(**plot_params, ci="sd", capsize=0.05)
plt.ylabel('')
plt.legend(loc='best', fontsize=12)
annotator = Annotator(syn3, pairs, **plot_params)
annotator.new_plot(syn3, pairs, plot="barplot", **plot_params)
annotator.configure(hide_non_significant=False, test="t-test_ind", text_format="star", loc="inside")
annotator.apply_test().annotate(line_offset_to_group=0.2, line_offset=0.1)
plt.show()

