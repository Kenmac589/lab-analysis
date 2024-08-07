import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame as df

# from scipy import stats as st
from statannotations.Annotator import Annotator


def condition_add(input_df, file_list, condition, limb, perturbation_state):
    for i in range(len(file_list)):
        limb = limb
        perturbation_state = perturbation_state
        mos_values = pd.read_csv(file_list[i], header=None)
        mos_values = mos_values.to_numpy()
        mos_values = mos_values.ravel()
        for j in range(len(mos_values)):
            entry = mos_values[j]
            if entry < 0.0:
                print(f"File with negative detected: {file_list[i]} with value {entry}")

            mos_entry = [[condition, limb, perturbation_state, entry]]

            input_df = input_df._append(
                pd.DataFrame(
                    mos_entry,
                    columns=["Condition", "Limb", "Perturbation State", "MoS (cm)"],
                ),
                ignore_index=True,
            )

    return input_df


dtrpre_non_lmos = [
    "./dtr_data/predtx/predtx_2non_lmos.csv",
    "./dtr_data/predtx/predtx_3non_lmos.csv",
    "./dtr_data/predtx/predtx_5non_lmos.csv",
    "./dtr_data/predtx/predtx_6non_lmos.csv",
    "./dtr_data/predtx/predtx_7non_lmos.csv",
]

dtrpre_non_rmos = [
    "./dtr_data/predtx/predtx_2non_rmos.csv",
    "./dtr_data/predtx/predtx_3non_rmos.csv",
    "./dtr_data/predtx/predtx_5non_rmos.csv",
    "./dtr_data/predtx/predtx_6non_rmos.csv",
    "./dtr_data/predtx/predtx_7non_rmos.csv",
]


dtrpre_per_lmos = [
    "./dtr_data/predtx/predtx_2per_lmos.csv",
    "./dtr_data/predtx/predtx_3per_lmos.csv",
    "./dtr_data/predtx/predtx_5per_lmos-1.csv",
    "./dtr_data/predtx/predtx_5per_lmos-2.csv",
    "./dtr_data/predtx/predtx_6per_lmos.csv",
    "./dtr_data/predtx/predtx_7per_lmos.csv",
]

dtrpre_per_rmos = [
    "./dtr_data/predtx/predtx_2per_rmos.csv",
    "./dtr_data/predtx/predtx_3per_rmos.csv",
    "./dtr_data/predtx/predtx_5per_rmos-1.csv",
    "./dtr_data/predtx/predtx_5per_rmos-2.csv",
    "./dtr_data/predtx/predtx_6per_rmos.csv",
    "./dtr_data/predtx/predtx_7per_rmos.csv",
]

dtrpre_sin_lmos = [
    "./dtr_data/predtx/predtx_2sin_lmos.csv",
    "./dtr_data/predtx/predtx_3sin_lmos-1.csv",
    "./dtr_data/predtx/predtx_3sin_lmos-2.csv",
    "./dtr_data/predtx/predtx_5sin_lmos.csv",
    "./dtr_data/predtx/predtx_6sin_lmos.csv",
    "./dtr_data/predtx/predtx_7sin_lmos.csv",
]

dtrpre_sin_rmos = [
    "./dtr_data/predtx/predtx_2sin_rmos.csv",
    "./dtr_data/predtx/predtx_3sin_rmos-1.csv",
    "./dtr_data/predtx/predtx_3sin_rmos-2.csv",
    "./dtr_data/predtx/predtx_5sin_rmos.csv",
    "./dtr_data/predtx/predtx_6sin_rmos.csv",
    "./dtr_data/predtx/predtx_7sin_rmos.csv",
]

dtrpost_non_lmos = [
    "./dtr_data/postdtx/postdtx_2non_lmos.csv",
    "./dtr_data/postdtx/postdtx_3non_lmos.csv",
    "./dtr_data/postdtx/postdtx_5non_lmos.csv",
    "./dtr_data/postdtx/postdtx_6non_lmos.csv",
]

dtrpost_non_rmos = [
    "./dtr_data/postdtx/postdtx_2non_rmos.csv",
    "./dtr_data/postdtx/postdtx_3non_rmos.csv",
    "./dtr_data/postdtx/postdtx_5non_rmos.csv",
    "./dtr_data/postdtx/postdtx_6non_rmos.csv",
]

dtrpost_per_lmos = [
    "./dtr_data/postdtx/postdtx_2per_lmos.csv",
    "./dtr_data/postdtx/postdtx_3per_lmos.csv",
    "./dtr_data/postdtx/postdtx_5per_lmos-1.csv",
    "./dtr_data/postdtx/postdtx_5per_lmos-2.csv",
    "./dtr_data/postdtx/postdtx_6per_lmos-auto.csv",
]

dtrpost_per_rmos = [
    "./dtr_data/postdtx/postdtx_2per_rmos.csv",
    "./dtr_data/postdtx/postdtx_3per_rmos.csv",
    "./dtr_data/postdtx/postdtx_5per_rmos-1.csv",
    "./dtr_data/postdtx/postdtx_5per_rmos-2.csv",
    "./dtr_data/postdtx/postdtx_6per_rmos-auto.csv",
]

dtrpost_sin_lmos = [
    "./dtr_data/postdtx/postdtx_2sin_lmos.csv",
    # "./dtr_data/postdtx/postdtx_3sin_lmos.csv",
    "./dtr_data/postdtx/postdtx_5sin_lmos.csv",
    "./dtr_data/postdtx/postdtx_6sin_lmos-man.csv",
]

dtrpost_sin_rmos = [
    "./dtr_data/postdtx/postdtx_2sin_rmos.csv",
    # "./dtr_data/postdtx/postdtx_3sin_rmos.csv",
    "./dtr_data/postdtx/postdtx_5sin_rmos.csv",
    "./dtr_data/postdtx/postdtx_6sin_rmos-man.csv",
]

mos_df = df(columns=["Condition", "Limb", "Perturbation State", "MoS (cm)"])

mos_df = condition_add(mos_df, dtrpre_non_lmos, "Pre-DTX", "Left", "Non-Perturbation")
mos_df = condition_add(mos_df, dtrpre_non_rmos, "Pre-DTX", "Right", "Non-Perturbation")
mos_df = condition_add(mos_df, dtrpre_per_lmos, "Pre-DTX", "Left", "Perturbation")
mos_df = condition_add(mos_df, dtrpre_per_rmos, "Pre-DTX", "Right", "Perturbation")
mos_df = condition_add(mos_df, dtrpre_sin_lmos, "Pre-DTX", "Left", "Sinusoidal")
mos_df = condition_add(mos_df, dtrpre_sin_rmos, "Pre-DTX", "Right", "Sinusoidal")

mos_df = condition_add(mos_df, dtrpost_non_lmos, "Post-DTX", "Left", "Non-Perturbation")
mos_df = condition_add(
    mos_df, dtrpost_non_rmos, "Post-DTX", "Right", "Non-Perturbation"
)
mos_df = condition_add(mos_df, dtrpost_per_lmos, "Post-DTX", "Left", "Perturbation")
mos_df = condition_add(mos_df, dtrpost_per_rmos, "Post-DTX", "Right", "Perturbation")
mos_df = condition_add(mos_df, dtrpost_sin_lmos, "Post-DTX", "Left", "Sinusoidal")
mos_df = condition_add(mos_df, dtrpost_sin_rmos, "Post-DTX", "Right", "Sinusoidal")

# For just comparing between perturbation
mos_combo = mos_df.drop(columns=["Limb"])
con_mos_combo = mos_df.drop(columns=["Limb"])

con_mos_combo.to_csv("./mos_limbs_combined_dtr.csv")

# Plotting
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set(
    style="white", font="serif", font_scale=1.8, palette="colorblind", rc=custom_params
)


combo_pairs = [
    [("Non-Perturbation"), ("Perturbation")],
]
combo_legend = ["Non-Perturbation", "Perturbation", "Sinusoidal"]

perturbation_state_order = ["Non-Perturbation", "Perturbation", "Sinusoidal"]

# plt.title("MoS between WT, Egr3 KO, and DTX Mice Pre and Post Injection")

# Intercondition Comparison
condition_pairs = [
    # Comparison within Pre-DTX condition
    [("Pre-DTX", "Non-Perturbation"), ("Pre-DTX", "Perturbation")],
    [("Pre-DTX", "Sinusoidal"), ("Pre-DTX", "Perturbation")],
    [("Pre-DTX", "Non-Perturbation"), ("Pre-DTX", "Sinusoidal")],
    # Comparison within Post-DTX condition
    [("Post-DTX", "Non-Perturbation"), ("Post-DTX", "Perturbation")],
    [("Post-DTX", "Sinusoidal"), ("Post-DTX", "Perturbation")],
    [("Post-DTX", "Non-Perturbation"), ("Post-DTX", "Sinusoidal")],
    # Comparison Between DTX conditions
    [("Pre-DTX", "Non-Perturbation"), ("Post-DTX", "Non-Perturbation")],
    [("Pre-DTX", "Perturbation"), ("Post-DTX", "Perturbation")],
    [("Pre-DTX", "Sinusoidal"), ("Post-DTX", "Sinusoidal")],
]

perturbation_state_order = ["Non-Perturbation", "Perturbation", "Sinusoidal"]
cond_combo_plot_params = {
    "data": con_mos_combo,
    "x": "Condition",
    "y": "MoS (cm)",
    "hue": "Perturbation State",
    "hue_order": perturbation_state_order,
    "inner": "point",
}

# axs[0].set_title("MoS between conditions")
cond_combo_comp = sns.violinplot(**cond_combo_plot_params, ci=95, capsize=0.05)
plt.legend(loc="upper left", fontsize=16)
plt.axhline(0, color="r")
annotator = Annotator(cond_combo_comp, condition_pairs, **cond_combo_plot_params)
annotator.new_plot(
    cond_combo_comp, condition_pairs, plot="violinplot", **cond_combo_plot_params
)
annotator.configure(
    hide_non_significant=True, test="t-test_ind", text_format="star", loc="inside"
)

annotator.apply_test().annotate(line_offset_to_group=0.2, line_offset=0.1)

fig = mpl.pyplot.gcf()
fig.set_size_inches(19.8, 10.80)
fig.tight_layout()
plt.savefig("./combined_figures/dtr_mos_violin_jitter.png", dpi=300)
# plt.show()
