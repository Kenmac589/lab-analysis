import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame as df

# from scipy import stats as st
from statannotations.Annotator import Annotator


def condition_add(
    input_df, file_list, condition, limb, perturbation_state, print_neg=True
):
    for i in range(len(file_list)):
        limb = limb
        perturbation_state = perturbation_state
        mos_values = pd.read_csv(file_list[i], header=None)
        mos_values = mos_values.to_numpy()
        mos_values = mos_values.ravel()
        # print(f"Average MoS for file {file_list[i]}: {np.mean(mos_values)}")
        for j in range(len(mos_values)):
            entry = mos_values[j]
            if print_neg is True:
                if entry < 0.0:
                    print(
                        f"File with negative detected: {file_list[i]} with value {entry}"
                    )

            mos_entry = [[condition, limb, perturbation_state, entry]]

            input_df = input_df._append(
                pd.DataFrame(
                    mos_entry,
                    columns=["Condition", "Limb", "Perturbation State", "MoS (cm)"],
                ),
                ignore_index=True,
            )

    return input_df


# Setting some info at beginning
save_fig_and_df = False
df_filename = "./aging_mos_4m_12m_18m_dtr.csv"
figure_title = "MoS between 4, 12 and 18 month old DTX Mice Pre and Post Injection"
figure_filename = "./aging/aging_for_grant/aging_mos_all_groups.svg"

wt_non_lmos = [
    "./wt_data/wt1non_lmos.csv",
    "./wt_data/wt2non_lmos.csv",
    "./wt_data/wt3non_lmos.csv",
    "./wt_data/wt4non_lmos.csv",
    "./wt_data/wt5non_lmos.csv",
]

wt_non_rmos = [
    "./wt_data/wt1non_rmos.csv",
    "./wt_data/wt2non_rmos.csv",
    "./wt_data/wt3non_rmos.csv",
    "./wt_data/wt4non_rmos.csv",
    "./wt_data/wt5non_rmos.csv",
]

wt_per_lmos = [
    "./wt_data/wt1per_lmos.csv",
    "./wt_data/wt2per_lmos.csv",
    "./wt_data/wt3per_lmos.csv",
    "./wt_data/wt4per_lmos.csv",
    "./wt_data/wt5per_lmos.csv",
]

wt_per_rmos = [
    "./wt_data/wt1per_rmos.csv",
    "./wt_data/wt2non_rmos.csv",
    "./wt_data/wt3per_rmos.csv",
    "./wt_data/wt4per_rmos.csv",
    "./wt_data/wt5per_rmos.csv",
]

wt_sin_lmos = [
    "./wt_data/wt1sin_lmos.csv",
    "./wt_data/wt2sin_lmos.csv",
    "./wt_data/wt3sin_lmos.csv",
    "./wt_data/wt4sin_lmos.csv",
]

wt_sin_rmos = [
    "./wt_data/wt1sin_rmos.csv",
    "./wt_data/wt2sin_rmos.csv",
    "./wt_data/wt3sin_rmos.csv",
    "./wt_data/wt4sin_rmos.csv",
]

# For Egr3
egr3_non_lmos = [
    "./egr3_data/egr3_6non_lmos.csv",
    "./egr3_data/egr3_7non_lmos.csv",
    "./egr3_data/egr3_8non_lmos.csv",
    "./egr3_data/egr3_9non_lmos.csv",
    "./egr3_data/egr3_10non_lmos.csv",
]

egr3_non_rmos = [
    "./egr3_data/egr3_6non_rmos.csv",
    "./egr3_data/egr3_7non_rmos.csv",
    "./egr3_data/egr3_8non_rmos.csv",
    "./egr3_data/egr3_9non_rmos.csv",
    "./egr3_data/egr3_10non_rmos.csv",
]

egr3_per_lmos = [
    "./egr3_data/egr3_6per_lmos.csv",
    "./egr3_data/egr3_7per_lmos.csv",
    "./egr3_data/egr3_8per_lmos.csv",
    "./egr3_data/egr3_9per_lmos-1.csv",
    "./egr3_data/egr3_9per_lmos-2.csv",
    "./egr3_data/egr3_10per_lmos-1.csv",
    "./egr3_data/egr3_10per_lmos-2.csv",
]

egr3_per_rmos = [
    "./egr3_data/egr3_6per_rmos.csv",
    "./egr3_data/egr3_7per_rmos.csv",
    "./egr3_data/egr3_8per_rmos.csv",
    "./egr3_data/egr3_9per_rmos-1.csv",
    "./egr3_data/egr3_9per_rmos-2.csv",
    "./egr3_data/egr3_10per_rmos-1.csv",
    "./egr3_data/egr3_10per_rmos-2.csv",
]

egr3_sin_lmos = [
    "./egr3_data/egr3_6sin_lmos.csv",
    "./egr3_data/egr3_7sin_lmos.csv",
    "./egr3_data/egr3_8sin_lmos.csv",
    "./egr3_data/egr3_9sin_lmos-1.csv",
    "./egr3_data/egr3_9sin_lmos-2.csv",
    "./egr3_data/egr3_10sin_lmos.csv",
]

egr3_sin_rmos = [
    "./egr3_data/egr3_6sin_rmos.csv",
    "./egr3_data/egr3_7sin_rmos.csv",
    "./egr3_data/egr3_8sin_rmos.csv",
    "./egr3_data/egr3_9sin_rmos-1.csv",
    "./egr3_data/egr3_9sin_rmos-2.csv",
    "./egr3_data/egr3_10sin_rmos.csv",
]


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

# 12m Mice
age_predtx_non_lmos = [
    "./aging/12mo/aging-12mo-saved_values/12mo-dtr_norosa-predtx-m1-non-lmos-0.csv",
    "./aging/12mo/aging-12mo-saved_values/12mo-dtr_norosa-predtx-m2-non-lmos-00.csv",
]

age_predtx_non_rmos = [
    "./aging/12mo/aging-12mo-saved_values/12mo-dtr_norosa-predtx-m1-non-rmos-0.csv",
    "./aging/12mo/aging-12mo-saved_values/12mo-dtr_norosa-predtx-m2-non-rmos-00.csv",
]
age_predtx_per_lmos = [
    "./aging/12mo/aging-12mo-saved_values/12mo-dtr_norosa-predtx-m1-per-lmos-9.csv",
    "./aging/12mo/aging-12mo-saved_values/12mo-dtr_norosa-predtx-m2-per-lmos-16.csv",
]
age_predtx_per_rmos = [
    "./aging/12mo/aging-12mo-saved_values/12mo-dtr_norosa-predtx-m1-per-rmos-9.csv",
    "./aging/12mo/aging-12mo-saved_values/12mo-dtr_norosa-predtx-m2-per-rmos-16.csv",
]
age_predtx_sin_lmos = [
    "./aging/12mo/aging-12mo-saved_values/12mo-dtr_norosa-predtx-m1-sin-lmos-18.csv",
    "./aging/12mo/aging-12mo-saved_values/12mo-dtr_norosa-predtx-m2-sin-lmos-19.csv",
]
age_predtx_sin_rmos = [
    "./aging/12mo/aging-12mo-saved_values/12mo-dtr_norosa-predtx-m1-sin-rmos-18.csv",
    "./aging/12mo/aging-12mo-saved_values/12mo-dtr_norosa-predtx-m2-sin-rmos-19.csv",
]

# 12m Post-DTX
age_postdtx_non_lmos = [
    "./aging/12mo/aging-12mo-saved_values/12mo-dtr_norosa-postdtx-m1-non-lmos-0.csv",
    "./aging/12mo/aging-12mo-saved_values/12mo-dtr_norosa-postdtx-m2-non-lmos-02.csv",
]

age_postdtx_non_rmos = [
    "./aging/12mo/aging-12mo-saved_values/12mo-dtr_norosa-postdtx-m1-non-rmos-0.csv",
    "./aging/12mo/aging-12mo-saved_values/12mo-dtr_norosa-postdtx-m2-non-rmos-02.csv",
]
age_postdtx_per_lmos = [
    "./aging/12mo/aging-12mo-saved_values/12mo-dtr_norosa-postdtx-m1-per-lmos-7.csv",
    "./aging/12mo/aging-12mo-saved_values/12mo-dtr_norosa-postdtx-m2-per-lmos-11.csv",
]
age_postdtx_per_rmos = [
    "./aging/12mo/aging-12mo-saved_values/12mo-dtr_norosa-postdtx-m1-per-rmos-7.csv",
    "./aging/12mo/aging-12mo-saved_values/12mo-dtr_norosa-postdtx-m2-per-rmos-11.csv",
]
age_postdtx_sin_lmos = [
    "./aging/12mo/aging-12mo-saved_values/12mo-dtr_norosa-postdtx-m1-sin-lmos-14.csv",
    "./aging/12mo/aging-12mo-saved_values/12mo-dtr_norosa-postdtx-m2-sin-lmos-18.csv",
]
age_postdtx_sin_rmos = [
    "./aging/12mo/aging-12mo-saved_values/12mo-dtr_norosa-postdtx-m1-sin-rmos-14.csv",
    "./aging/12mo/aging-12mo-saved_values/12mo-dtr_norosa-postdtx-m2-sin-rmos-18.csv",
]

# 18m Mouse
# M1: 01, 14, 20
old_predtx_non_lmos = [
    "./aging/18mo/aging-18mo-saved_values/18mo-rap-predtx-m1-non-lmos-00.csv",
]

old_predtx_non_rmos = [
    "./aging/18mo/aging-18mo-saved_values/18mo-rap-predtx-m1-non-rmos-00.csv",
]
old_predtx_per_lmos = [
    "./aging/18mo/aging-18mo-saved_values/18mo-rap-predtx-m1-per-lmos-14.csv",
]
old_predtx_per_rmos = [
    "./aging/18mo/aging-18mo-saved_values/18mo-rap-predtx-m1-per-rmos-14.csv",
]
old_predtx_sin_lmos = [
    "./aging/18mo/aging-18mo-saved_values/18mo-rap-predtx-m1-sin-lmos-20.csv",
]
old_predtx_sin_rmos = [
    "./aging/18mo/aging-18mo-saved_values/18mo-rap-predtx-m1-sin-rmos-20.csv",
]

mos_df = df(columns=["Condition", "Limb", "Perturbation State", "MoS (cm)"])

mos_df = condition_add(mos_df, wt_non_lmos, "WT", "Left", "Non-Perturbation")
mos_df = condition_add(mos_df, wt_non_rmos, "WT", "Right", "Non-Perturbation")
mos_df = condition_add(mos_df, wt_per_lmos, "WT", "Left", "Perturbation")
mos_df = condition_add(mos_df, wt_per_rmos, "WT", "Right", "Perturbation")
mos_df = condition_add(mos_df, wt_sin_lmos, "WT", "Left", "Sinusoidal")
mos_df = condition_add(mos_df, wt_sin_rmos, "WT", "Right", "Sinusoidal")

mos_df = condition_add(mos_df, egr3_non_lmos, "Egr3", "Left", "Non-Perturbation")
mos_df = condition_add(mos_df, egr3_non_rmos, "Egr3", "Right", "Non-Perturbation")
mos_df = condition_add(mos_df, egr3_per_lmos, "Egr3", "Left", "Perturbation")
mos_df = condition_add(mos_df, egr3_per_rmos, "Egr3", "Right", "Perturbation")
mos_df = condition_add(mos_df, egr3_sin_lmos, "Egr3", "Left", "Sinusoidal")
mos_df = condition_add(mos_df, egr3_sin_rmos, "Egr3", "Right", "Sinusoidal")

# Adding young DTR mice
mos_df = condition_add(
    mos_df, dtrpre_non_lmos, "4m Pre-DTX", "Left", "Non-Perturbation"
)
mos_df = condition_add(
    mos_df, dtrpre_non_rmos, "4m Pre-DTX", "Right", "Non-Perturbation"
)
mos_df = condition_add(mos_df, dtrpre_per_lmos, "4m Pre-DTX", "Left", "Perturbation")
mos_df = condition_add(mos_df, dtrpre_per_rmos, "4m Pre-DTX", "Right", "Perturbation")
mos_df = condition_add(mos_df, dtrpre_sin_lmos, "4m Pre-DTX", "Left", "Sinusoidal")
mos_df = condition_add(mos_df, dtrpre_sin_rmos, "4m Pre-DTX", "Right", "Sinusoidal")

mos_df = condition_add(
    mos_df, dtrpost_non_lmos, "4m Post-DTX", "Left", "Non-Perturbation"
)
mos_df = condition_add(
    mos_df, dtrpost_non_rmos, "4m Post-DTX", "Right", "Non-Perturbation"
)
mos_df = condition_add(mos_df, dtrpost_per_lmos, "4m Post-DTX", "Left", "Perturbation")
mos_df = condition_add(mos_df, dtrpost_per_rmos, "4m Post-DTX", "Right", "Perturbation")
mos_df = condition_add(mos_df, dtrpost_sin_lmos, "4m Post-DTX", "Left", "Sinusoidal")
mos_df = condition_add(mos_df, dtrpost_sin_rmos, "4m Post-DTX", "Right", "Sinusoidal")

# Adding 12mo aged DTR mice
mos_df = condition_add(
    mos_df, age_predtx_non_lmos, "12m Pre-DTX", "Left", "Non-Perturbation"
)
mos_df = condition_add(
    mos_df, age_predtx_non_rmos, "12m Pre-DTX", "Right", "Non-Perturbation"
)
mos_df = condition_add(
    mos_df, age_predtx_per_lmos, "12m Pre-DTX", "Left", "Perturbation"
)
mos_df = condition_add(
    mos_df, age_predtx_per_rmos, "12m Pre-DTX", "Right", "Perturbation"
)
mos_df = condition_add(mos_df, age_predtx_sin_lmos, "12m Pre-DTX", "Left", "Sinusoidal")
mos_df = condition_add(
    mos_df, age_predtx_sin_rmos, "12m Pre-DTX", "Right", "Sinusoidal"
)

# Adding 12m Post-DTX group
mos_df = condition_add(
    mos_df, age_postdtx_non_lmos, "12m Post-DTX", "Left", "Non-Perturbation"
)
mos_df = condition_add(
    mos_df, age_postdtx_non_rmos, "12m Post-DTX", "Right", "Non-Perturbation"
)
mos_df = condition_add(
    mos_df, age_postdtx_per_lmos, "12m Post-DTX", "Left", "Perturbation"
)
mos_df = condition_add(
    mos_df, age_postdtx_per_rmos, "12m Post-DTX", "Right", "Perturbation"
)
mos_df = condition_add(
    mos_df, age_postdtx_sin_lmos, "12m Post-DTX", "Left", "Sinusoidal"
)
mos_df = condition_add(
    mos_df, age_postdtx_sin_rmos, "12m Post-DTX", "Right", "Sinusoidal"
)

# Adding 18mo old group
mos_df = condition_add(
    mos_df, old_predtx_non_lmos, "18m Pre-DTX", "Left", "Non-Perturbation"
)
mos_df = condition_add(
    mos_df, old_predtx_non_rmos, "18m Pre-DTX", "Right", "Non-Perturbation"
)
mos_df = condition_add(
    mos_df, old_predtx_per_lmos, "18m Pre-DTX", "Left", "Perturbation"
)
mos_df = condition_add(
    mos_df, old_predtx_per_rmos, "18m Pre-DTX", "Right", "Perturbation"
)
mos_df = condition_add(mos_df, old_predtx_sin_lmos, "18m Pre-DTX", "Left", "Sinusoidal")
mos_df = condition_add(
    mos_df, old_predtx_sin_rmos, "18m Pre-DTX", "Right", "Sinusoidal"
)
# Aged HL only
# mos_df = condition_add(
#     mos_df, age_predtx_non_lmos_hl, "12m Pre-DTX HL", "Left", "Non-Perturbation"
# )
# mos_df = condition_add(
#     mos_df, age_predtx_non_rmos, "12m Pre-DTX HL", "Right", "Non-Perturbation"
# )
# mos_df = condition_add(
#     mos_df, age_predtx_sin_lmos, "12m Pre-DTX HL", "Left", "Sinusoidal"
# )
# mos_df = condition_add(
#     mos_df, age_predtx_sin_rmos, "12m Pre-DTX HL", "Right", "Sinusoidal"
# )

# For just comparing between perturbation
mos_combo = mos_df.drop(columns=["Limb"])
con_mos_combo = mos_df.drop(columns=["Limb"])

# Plotting
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set(
    style="white", font="serif", font_scale=1.6, palette="colorblind", rc=custom_params
)


combo_pairs = [
    [("Non-Perturbation"), ("Perturbation")],
]
combo_legend = ["Non-Perturbation", "Perturbation", "Sinusoidal"]

perturbation_state_order = ["Non-Perturbation", "Perturbation", "Sinusoidal"]

# Only Non and Sinusoidal
# -----------------------
# fig, axs = plt.subplots(1, 2)
# combo_legend = ["Non-Perturbation", "Sinusoidal"]

# perturbation_state_order = ["Non-Perturbation", "Sinusoidal"]
# -----------------------

plt.title(figure_title)

# Intercondition Comparison
condition_pairs = [
    # Comparison within wildtype condition
    [("WT", "Non-Perturbation"), ("WT", "Perturbation")],
    [("WT", "Sinusoidal"), ("WT", "Perturbation")],
    [("WT", "Non-Perturbation"), ("WT", "Sinusoidal")],
    # # Comparison between Wildtype and 4m Pre-DTX
    [("WT", "Non-Perturbation"), ("4m Pre-DTX", "Non-Perturbation")],
    [("WT", "Sinusoidal"), ("4m Pre-DTX", "Sinusoidal")],
    [("WT", "Perturbation"), ("4m Pre-DTX", "Perturbation")],
    # # Comparison within Egr3 condition
    [("Egr3", "Non-Perturbation"), ("Egr3", "Perturbation")],
    [("Egr3", "Sinusoidal"), ("Egr3", "Perturbation")],
    [("Egr3", "Non-Perturbation"), ("Egr3", "Sinusoidal")],
    # Comparison within 4m Pre-DTX condition
    [("4m Pre-DTX", "Non-Perturbation"), ("4m Pre-DTX", "Perturbation")],
    [("4m Pre-DTX", "Sinusoidal"), ("4m Pre-DTX", "Perturbation")],
    [("4m Pre-DTX", "Non-Perturbation"), ("4m Pre-DTX", "Sinusoidal")],
    # Comparison within 4m Post-DTX condition
    [("4m Post-DTX", "Non-Perturbation"), ("4m Post-DTX", "Perturbation")],
    [("4m Post-DTX", "Sinusoidal"), ("4m Post-DTX", "Perturbation")],
    [("4m Post-DTX", "Non-Perturbation"), ("4m Post-DTX", "Sinusoidal")],
    # Comparison within 12m Pre-DTX condition
    [("12m Pre-DTX", "Non-Perturbation"), ("12m Pre-DTX", "Perturbation")],
    [("12m Pre-DTX", "Sinusoidal"), ("12m Pre-DTX", "Perturbation")],
    [("12m Pre-DTX", "Non-Perturbation"), ("12m Pre-DTX", "Sinusoidal")],
    # Comparison within 12m Post-DTX condition
    [("12m Post-DTX", "Non-Perturbation"), ("12m Post-DTX", "Perturbation")],
    [("12m Post-DTX", "Sinusoidal"), ("12m Post-DTX", "Perturbation")],
    [("12m Post-DTX", "Non-Perturbation"), ("12m Post-DTX", "Sinusoidal")],
    # Comparison between 4m and 12m DTR mice
    [("12m Pre-DTX", "Non-Perturbation"), ("4m Pre-DTX", "Non-Perturbation")],
    [("12m Pre-DTX", "Sinusoidal"), ("4m Pre-DTX", "Sinusoidal")],
    [("12m Pre-DTX", "Perturbation"), ("4m Pre-DTX", "Perturbation")],
    # Comparison between 18m and 12m DTR mice
    [("12m Pre-DTX", "Non-Perturbation"), ("18m Pre-DTX", "Non-Perturbation")],
    [("12m Pre-DTX", "Sinusoidal"), ("18m Pre-DTX", "Sinusoidal")],
    [("12m Pre-DTX", "Perturbation"), ("18m Pre-DTX", "Perturbation")],
    # Comparison within 18m DTR mice
    [("18m Pre-DTX", "Non-Perturbation"), ("18m Pre-DTX", "Perturbation")],
    [("18m Pre-DTX", "Sinusoidal"), ("18m Pre-DTX", "Perturbation")],
    [("18m Pre-DTX", "Non-Perturbation"), ("18m Pre-DTX", "Sinusoidal")],
    # Comparison between Wildtype and 18m Pre-DTX
    [("WT", "Non-Perturbation"), ("18m Pre-DTX", "Non-Perturbation")],
    [("WT", "Sinusoidal"), ("18m Pre-DTX", "Sinusoidal")],
    [("WT", "Perturbation"), ("18m Pre-DTX", "Perturbation")],
    # Comparing Hindlimb Only within 12m
    # [("12m Pre-DTX", "Non-Perturbation"), ("12m Pre-DTX HL", "Non-Perturbation")],
    # [("12m Pre-DTX", "Sinusoidal"), ("12m Pre-DTX HL", "Sinusoidal")],
    # [("12m Pre-DTX HL", "Non-Perturbation"), ("12m Pre-DTX HL", "Sinusoidal")],
]

fig = mpl.pyplot.gcf()
fig.set_size_inches(19.8, 10.80)
fig.tight_layout()

perturbation_state_order = ["Non-Perturbation", "Perturbation", "Sinusoidal"]
# perturbation_state_order = ["Non-Perturbation", "Sinusoidal"]
cond_combo_plot_params = {
    "data": con_mos_combo,
    "x": "Condition",
    "y": "MoS (cm)",
    "hue": "Perturbation State",
    "hue_order": perturbation_state_order,
    # "inner": "point",
}

# axs[0].set_title("MoS between conditions")
cond_combo_comp = sns.barplot(**cond_combo_plot_params, ci=95, capsize=0.05)
plt.legend(loc="upper left", fontsize=16)
annotator = Annotator(cond_combo_comp, condition_pairs, **cond_combo_plot_params)
annotator.new_plot(
    cond_combo_comp, condition_pairs, plot="barplot", **cond_combo_plot_params
)
annotator.configure(
    hide_non_significant=True,
    test="t-test_ind",
    text_format="star",
    loc="inside",
)

annotator.apply_test().annotate(line_offset_to_group=0.2, line_offset=0.1)

fig = mpl.pyplot.gcf()
fig.set_size_inches(19.8, 10.80)
fig.tight_layout()

if save_fig_and_df is True:
    plt.savefig(figure_filename, dpi=300)
    con_mos_combo.to_csv(df_filename, index=False)
else:
    print("Results not saved")

# plt.show()
