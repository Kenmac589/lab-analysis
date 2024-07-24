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


def append_to_df(input_df, file_list, condition, limb):
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


pre_lmos = [
    "./wt-no-emg/wt-no-emg-m1/wt-no-emg-m1-pre/wt-no-emg-m1-non-pre-lmos.csv",
    "./wt-no-emg/wt-no-emg-m2/wt-no-emg-m2-pre/wt-no-emg-m2-non-pre-lmos-1.csv",
    "./wt-no-emg/wt-no-emg-m2/wt-no-emg-m2-pre/wt-no-emg-m2-non-pre-lmos-2.csv",
]

pre_rmos = [
    "./wt-no-emg/wt-no-emg-m1/wt-no-emg-m1-pre/wt-no-emg-m1-non-pre-rmos.csv",
    "./wt-no-emg/wt-no-emg-m2/wt-no-emg-m2-pre/wt-no-emg-m2-non-pre-rmos-1.csv",
    "./wt-no-emg/wt-no-emg-m2/wt-no-emg-m2-pre/wt-no-emg-m2-non-pre-rmos-2.csv",
]

post_lmos = [
    "./lr-walking/rdir/rwalk-1-lmos.csv",
    "./lr-walking/rdir/rwalk-2-lmos.csv",
    "./lr-walking/rdir/rwalk-3-lmos.csv",
    "./lr-walking/rdir/rwalk-4-lmos.csv",
    "./lr-walking/rdir/rwalk-5-lmos.csv",
]

post_rmos = [
    "./lr-walking/rdir/rwalk-1-rmos.csv",
    "./lr-walking/rdir/rwalk-2-rmos.csv",
    "./lr-walking/rdir/rwalk-3-rmos.csv",
    "./lr-walking/rdir/rwalk-4-rmos.csv",
    "./lr-walking/rdir/rwalk-5-rmos.csv",
]

mos_df = df(columns=["Condition", "Limb", "MoS (cm)"])
mos_diff_df = df(columns=["Condition", "LR Difference"])

mos_df = condition_add(mos_df, pre_lmos, "Pre-Implant", "Left")
mos_df = condition_add(mos_df, pre_rmos, "Pre-Implant", "Right")
mos_df = condition_add(mos_df, post_lmos, "Post-Implant", "Left")
mos_df = condition_add(mos_df, post_rmos, "Post-Implant", "Right")

# For just comparing between perturbation
mos_limb_diff_pre = mos_df[mos_df["Condition"] == "Pre-Implant"]
mos_limb_diff_pre_left = mos_limb_diff_pre[mos_limb_diff_pre["Limb"] == "Left"]
mos_pre_left = mos_limb_diff_pre_left["MoS (cm)"].to_numpy(dtype=float)
mos_limb_diff_pre_right = mos_limb_diff_pre[mos_limb_diff_pre["Limb"] == "Right"]
mos_pre_right = mos_limb_diff_pre_right["MoS (cm)"].to_numpy(dtype=float)

max_length = 0

if len(mos_pre_left) > len(mos_pre_right):
    max_length = len(mos_pre_right)
else:
    max_length = len(mos_pre_left)
mos_pre_diff = np.abs(mos_pre_left[:max_length] - mos_pre_right[:max_length])

for i in range(len(mos_pre_diff)):
    condition = "Pre-Implant"
    fixed_array = mos_pre_diff.ravel()
    diff_entry = [[condition, fixed_array[i]]]
    mos_diff_df = mos_diff_df._append(
        pd.DataFrame(diff_entry, columns=["Condition", "LR Difference"]),
        ignore_index=True,
    )

mos_limb_diff_post = mos_df[mos_df["Condition"] == "Post-Implant"]
mos_limb_diff_post_left = mos_limb_diff_post[mos_limb_diff_post["Limb"] == "Left"]
mos_post_left = mos_limb_diff_post_left["MoS (cm)"].to_numpy(dtype=float)
mos_limb_diff_post_right = mos_limb_diff_post[mos_limb_diff_post["Limb"] == "Right"]
mos_post_right = mos_limb_diff_post_right["MoS (cm)"].to_numpy(dtype=float)

if len(mos_post_left) > len(mos_post_right):
    max_length = len(mos_post_right)
else:
    max_length = len(mos_post_left)

mos_post_diff = np.abs(mos_post_left[:max_length] - mos_post_right[:max_length])

for i in range(len(mos_post_diff)):
    condition = "Post-Implant"
    fixed_array = mos_post_diff.ravel()
    diff_entry = [[condition, fixed_array[i]]]
    mos_diff_df = mos_diff_df._append(
        pd.DataFrame(diff_entry, columns=["Condition", "LR Difference"]),
        ignore_index=True,
    )

print(mos_diff_df)


mos_combo = mos_df.drop(columns=["Limb"])


# Comparing gap


# Plotting
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set(
    style="white", font="serif", font_scale=1.8, palette="colorblind", rc=custom_params
)


fig, axs = plt.subplots(1, 2)

limb_pairs = [
    # Within Conditions between Limb
    [("Right", "Pre-Implant"), ("Left", "Pre-Implant")],
    [("Right", "Post-Implant"), ("Left", "Post-Implant")],
    # Between Conditions with same limb
    [("Right", "Pre-Implant"), ("Right", "Post-Implant")],
    [("Left", "Pre-Implant"), ("Left", "Post-Implant")],
]

combo_pairs = [
    [("Pre-Implant"), ("Post-Implant")],
]
combo_legend = ["Pre-Implant", "Post-Implant"]

perturbation_state_order = ["Pre-Implant", "Post-Implant"]

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

# axs[1].set_title("Average Difference between L and R limbs")
# axs[1].set_ylim(0, 0.6)
# sns.barplot(mos_limb_diff, ax=axs[1])
# Total MoS for conditions
combo_plot_params = {
    "data": mos_diff_df,
    "x": "Condition",
    "y": "LR Difference",
}

# axs[1].set_title("MoS for WT by Pertubation State")
combo_comp = sns.barplot(**combo_plot_params, ci=95, capsize=0.05, ax=axs[1])
# axs[1].legend(combo_legend, loc="upper left", fontsize=12)
annotator = Annotator(combo_comp, combo_pairs, **combo_plot_params)
annotator.new_plot(combo_comp, combo_pairs, plot="barplot", **combo_plot_params)
annotator.configure(
    hide_non_significant=False, test="t-test_ind", text_format="star", loc="inside"
)
annotator.apply_test().annotate(line_offset_to_group=0.2, line_offset=0.1)


fig = plt.gcf()
fig.set_size_inches(19.8, 10.80)
fig.tight_layout()
# plt.savefig("./wt-no-emg/wt-no-emg-mos-auto.svg", dpi=300)
plt.show()
