import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame as df
# from scipy import stats as st
from statannotations.Annotator import Annotator


def condition_add(input_df, file_list, limb, perturbation_state):
    for i in range(len(file_list)):
        limb = limb
        perturbation_state = perturbation_state
        mos_values = pd.read_csv(file_list[i], header=None)
        mos_values = mos_values.to_numpy()
        mos_values = mos_values.ravel()
        for j in range(len(mos_values)):
            entry = mos_values[j]
            mos_entry = [[limb, perturbation_state, entry]]

            input_df = input_df._append(
                pd.DataFrame(mos_entry, columns=["Limb", "Perturbation State", "MoS"]),
                ignore_index=True,
            )

    return input_df


wt_non_lmos = [
    "./wt1non_lmos.csv",
    "./wt2non_lmos.csv",
    "./wt3non_lmos.csv",
    "./wt4non_lmos.csv",
    "./wt5non_lmos.csv",
]

wt_non_rmos = [
    "./wt1non_rmos.csv",
    "./wt2non_rmos.csv",
    "./wt3non_rmos.csv",
    "./wt4non_rmos.csv",
    "./wt5non_rmos.csv",
]

wt_per_lmos = [
    "./wt1per_lmos.csv",
    "./wt2per_lmos.csv",
    "./wt3per_lmos.csv",
    "./wt4per_lmos.csv",
    "./wt5per_lmos.csv",
]

wt_per_rmos = [
    "./wt1per_rmos.csv",
    "./wt2non_rmos.csv",
    "./wt3per_rmos.csv",
    "./wt4per_rmos.csv",
    "./wt5per_rmos.csv",
]

wt_sin_lmos = [
    "./wt1sin_lmos.csv",
    "./wt2sin_lmos.csv",
    "./wt3sin_lmos.csv",
    "./wt4sin_lmos.csv",
]

wt_sin_rmos = [
    "./wt1sin_rmos.csv",
    "./wt2sin_rmos.csv",
    "./wt3sin_rmos.csv",
    "./wt4sin_rmos.csv",
]


mos_df = df(columns=["Limb", "Perturbation State", "MoS"])

mos_df = condition_add(mos_df, wt_non_lmos, "Left", "Non-Perturbation")
mos_df = condition_add(mos_df, wt_non_rmos, "Right", "Non-Perturbation")
mos_df = condition_add(mos_df, wt_per_lmos, "Left", "Perturbation")
mos_df = condition_add(mos_df, wt_per_rmos, "Right", "Perturbation")
mos_df = condition_add(mos_df, wt_sin_lmos, "Left", "Sinusoidal")
mos_df = condition_add(mos_df, wt_sin_rmos, "Right", "Sinusoidal")

# For just comparing between perturbation
mos_combo = mos_df.drop(columns=["Limb"])

# Plotting
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set(style="white", font="Arial", font_scale=1.5, rc=custom_params)

fig, axs = plt.subplots(1, 2)

limb_pairs = [
    [("Left", "Non-Perturbation"), ("Left", "Perturbation")],
    [("Left", "Non-Perturbation"), ("Left", "Sinusoidal")],
    [("Right", "Non-Perturbation"), ("Right", "Perturbation")],
    [("Right", "Non-Perturbation"), ("Right", "Sinusoidal")],
]

combo_pairs = [
    [("Non-Perturbation"), ("Perturbation")],
]
combo_legend = ["Non-Perturbation", "Perturbation", "Sinusoidal"]

perturbation_state_order = ["Non-Perturbation", "Perturbation", "Sinusoidal"]

# Plot for limb comparison
limb_plot_params = {
    "data": mos_df,
    "x": "Limb",
    "y": "MoS",
    "hue": "Perturbation State",
    "hue_order": perturbation_state_order,
}

fig.suptitle("Margin of Lateral Dynamic Lateral Stability (MoS) in WT")

axs[0].set_title("MoS for WT by Limb")
limb_comp = sns.barplot(**limb_plot_params, ci="sd", capsize=0.05, ax=axs[0])
# axs[0].legend(fontsize=12, bbox_to_anchor=(2.49, 0.7))
axs[0].legend(fontsize=12, loc="best")
annotator = Annotator(limb_comp, limb_pairs, **limb_plot_params)
annotator.new_plot(limb_comp, limb_pairs, plot="barplot", **limb_plot_params)
annotator.configure(
    hide_non_significant=False, test="t-test_welch", text_format="star", loc="inside"
)
annotator.apply_test().annotate(line_offset_to_group=0.2, line_offset=0.1)

# Total MoS for conditions
combo_plot_params = {
    "data": mos_combo,
    "x": "Perturbation State",
    "y": "MoS",
}

axs[1].set_title("MoS for WT by Pertubation State")
combo_comp = sns.barplot(**combo_plot_params, ci="sd", capsize=0.05, ax=axs[1])
# axs[1].legend(combo_legend, loc="upper left", fontsize=12)
annotator = Annotator(combo_comp, combo_pairs, **combo_plot_params)
annotator.new_plot(combo_comp, combo_pairs, plot="barplot", **combo_plot_params)
annotator.configure(
    hide_non_significant=False, test="t-test_welch", text_format="star", loc="inside"
)
annotator.apply_test().annotate(line_offset_to_group=0.2, line_offset=0.1)

plt.show()