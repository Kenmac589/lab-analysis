import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame as df

# from scipy import stats as st
from statannotations.Annotator import Annotator


def condition_add(input_df, file_list, condition, limb):
    for i in range(len(file_list)):
        limb = limb
        mos_values = pd.read_csv(file_list[i], header=None)
        mos_values = mos_values.to_numpy()
        mos_values = mos_values.ravel()
        for j in range(len(mos_values)):
            entry = mos_values[j]
            mos_entry = [[condition, limb, entry]]

            input_df = input_df._append(
                pd.DataFrame(
                    mos_entry,
                    columns=["Condition", "Limb", "MoS (cm)"],
                ),
                ignore_index=True,
            )

    return input_df


lwalk_lmos = [
    "./lr-walking/file_string_test/lwalk-1-lmos.csv",
    "./lr-walking/file_string_test/lwalk-2-lmos.csv",
    "./lr-walking/file_string_test/lwalk-3-lmos.csv",
    "./lr-walking/file_string_test/lwalk-4-lmos.csv",
    "./lr-walking/file_string_test/lwalk-5-lmos.csv",
]

lwalk_rmos = [
    "./lr-walking/file_string_test/lwalk-1-rmos.csv",
    "./lr-walking/file_string_test/lwalk-2-rmos.csv",
    "./lr-walking/file_string_test/lwalk-3-rmos.csv",
    "./lr-walking/file_string_test/lwalk-4-rmos.csv",
    "./lr-walking/file_string_test/lwalk-5-rmos.csv",
]

rwalk_lmos = [
    "./lr-walking/file_string_test/rwalk-1-lmos.csv",
    "./lr-walking/file_string_test/rwalk-2-lmos.csv",
    "./lr-walking/file_string_test/rwalk-3-lmos.csv",
    "./lr-walking/file_string_test/rwalk-4-lmos.csv",
    "./lr-walking/file_string_test/rwalk-5-lmos.csv",
]

rwalk_rmos = [
    "./lr-walking/file_string_test/rwalk-1-rmos.csv",
    "./lr-walking/file_string_test/rwalk-2-rmos.csv",
    "./lr-walking/file_string_test/rwalk-3-rmos.csv",
    "./lr-walking/file_string_test/rwalk-4-rmos.csv",
    "./lr-walking/file_string_test/rwalk-5-rmos.csv",
]

mos_df = df(columns=["Condition", "Limb", "MoS (cm)"])

mos_df = condition_add(mos_df, lwalk_lmos, "Left Direction", "Left")
mos_df = condition_add(mos_df, lwalk_rmos, "Left Direction", "Right")
mos_df = condition_add(mos_df, rwalk_lmos, "Right Direction", "Left")
mos_df = condition_add(mos_df, rwalk_rmos, "Right Direction", "Right")

print(mos_df)
# For just comparing between perturbation
mos_combo = mos_df.drop(columns=["Limb"])

# Plotting
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set(
    style="white", font="serif", font_scale=1.8, palette="colorblind", rc=custom_params
)


fig, axs = plt.subplots(1, 2)

limb_pairs = [
    [("Right", "Left Direction"), ("Left", "Left Direction")],
    [("Right", "Right Direction"), ("Left", "Right Direction")],
]

combo_pairs = [
    [("Left Direction"), ("Right Direction")],
]
combo_legend = ["Left Direction", "Right Direction"]

perturbation_state_order = ["Left Direction", "Right Direction"]

# Plot for limb comparison
limb_plot_params = {
    "data": mos_df,
    "x": "Limb",
    "y": "MoS (cm)",
    "hue": "Condition",
    "hue_order": perturbation_state_order,
}

# fig.suptitle("Margin of Lateral Dynamic Lateral Stability (MoS) in WT")

# axs[0].set_title("MoS for WT by Limb")
limb_comp = sns.barplot(**limb_plot_params, ci=95, capsize=0.05, ax=axs[0])
# axs[0].legend(fontsize=12, bbox_to_anchor=(2.49, 0.7))
axs[0].legend(fontsize=16, loc="best")
annotator = Annotator(limb_comp, limb_pairs, **limb_plot_params)
annotator.new_plot(limb_comp, limb_pairs, plot="barplot", **limb_plot_params)
annotator.configure(
    hide_non_significant=False, test="t-test_ind", text_format="star", loc="inside"
)
annotator.apply_test().annotate(line_offset_to_group=0.2, line_offset=0.1)

# Total MoS for conditions
combo_plot_params = {
    "data": mos_combo,
    "x": "Condition",
    "y": "MoS (cm)",
    "inner": "point",
}

# axs[1].set_title("MoS for WT by Pertubation State")
combo_comp = sns.violinplot(**combo_plot_params, ci=95, capsize=0.05, ax=axs[1])
# axs[1].legend(combo_legend, loc="upper left", fontsize=12)
annotator = Annotator(combo_comp, combo_pairs, **combo_plot_params)
annotator.new_plot(combo_comp, combo_pairs, plot="violinplot", **combo_plot_params)
annotator.configure(
    hide_non_significant=False, test="t-test_ind", text_format="star", loc="inside"
)
annotator.apply_test().annotate(line_offset_to_group=0.2, line_offset=0.1)


fig = plt.gcf()
fig.set_size_inches(19.8, 10.80)
fig.tight_layout()
plt.savefig("./lr-walking/lr-walking_by_limb_auto-test.svg", dpi=300)
# plt.show()
