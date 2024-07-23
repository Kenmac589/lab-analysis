import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg
import scikit_posthocs as sp
import seaborn as sns
import statsmodels.api as sa
from pandas import DataFrame as df
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


def condition_add(input_df, file_list, condition, limb, perturbation_state):
    for i in range(len(file_list)):
        limb = limb
        perturbation_state = perturbation_state
        mos_values = pd.read_csv(file_list[i], header=None)
        mos_values = mos_values.to_numpy()
        mos_values = mos_values.ravel()
        for j in range(len(mos_values)):
            entry = mos_values[j]
            mos_entry = [[condition, limb, perturbation_state, entry]]

            input_df = input_df._append(
                pd.DataFrame(
                    mos_entry,
                    columns=["Condition", "Limb", "Perturbation_State", "MoS"],
                ),
                ignore_index=True,
            )

    return input_df


def main():

    wt_non_lmos = [
        "./wt_data/wt1non_lmos.csv",
        # "./wt_data/wt2non_lmos.csv",
        "./wt_data/wt3non_lmos.csv",
        "./wt_data/wt4non_lmos.csv",
        "./wt_data/wt5non_lmos.csv",
    ]

    wt_non_rmos = [
        "./wt_data/wt1non_rmos.csv",
        # "./wt_data/wt2non_rmos.csv",
        "./wt_data/wt3non_rmos.csv",
        "./wt_data/wt4non_rmos.csv",
        "./wt_data/wt5non_rmos.csv",
    ]

    wt_per_lmos = [
        # "./wt_data/wt1per_lmos.csv",
        # "./wt_data/wt2per_lmos.csv",
        "./wt_data/wt3per_lmos.csv",
        "./wt_data/wt4per_lmos.csv",
        "./wt_data/wt5per_lmos.csv",
    ]

    wt_per_rmos = [
        # "./wt_data/wt1per_rmos.csv",
        # "./wt_data/wt2non_rmos.csv",
        "./wt_data/wt3per_rmos.csv",
        "./wt_data/wt4per_rmos.csv",
        "./wt_data/wt5per_rmos.csv",
    ]

    wt_sin_lmos = [
        "./wt_data/wt1sin_lmos.csv",
        # "./wt_data/wt2sin_lmos.csv",
        "./wt_data/wt3sin_lmos.csv",
        "./wt_data/wt4sin_lmos.csv",
    ]

    wt_sin_rmos = [
        "./wt_data/wt1sin_rmos.csv",
        # "./wt_data/wt2sin_rmos.csv",
        "./wt_data/wt3sin_rmos.csv",
        "./wt_data/wt4sin_rmos.csv",
    ]

    # For Egr3
    egr3_non_lmos = [
        "./egr3_data/egr3_6non_lmos.csv",
        # "./egr3_data/egr3_7non_lmos.csv",
        "./egr3_data/egr3_8non_lmos.csv",
        "./egr3_data/egr3_9non_lmos.csv",
        "./egr3_data/egr3_10non_lmos.csv",
    ]

    egr3_non_rmos = [
        "./egr3_data/egr3_6non_rmos.csv",
        # "./egr3_data/egr3_7non_rmos.csv",
        "./egr3_data/egr3_8non_rmos.csv",
        "./egr3_data/egr3_9non_rmos.csv",
        "./egr3_data/egr3_10non_rmos.csv",
    ]

    egr3_per_lmos = [
        "./egr3_data/egr3_6per_lmos.csv",
        # "./egr3_data/egr3_7per_lmos.csv",
        # "./egr3_data/egr3_8per_lmos.csv",
        "./egr3_data/egr3_9per_lmos-1.csv",
        "./egr3_data/egr3_9per_lmos-2.csv",
        "./egr3_data/egr3_10per_lmos-1.csv",
        "./egr3_data/egr3_10per_lmos-2.csv",
    ]

    egr3_per_rmos = [
        "./egr3_data/egr3_6per_rmos.csv",
        # "./egr3_data/egr3_7per_rmos.csv",
        # "./egr3_data/egr3_8per_rmos.csv",
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

    mos_df = df(columns=["Condition", "Limb", "Perturbation_State", "MoS"])

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

    mos_df = condition_add(
        mos_df, dtrpre_non_lmos, "Pre-DTX", "Left", "Non-Perturbation"
    )
    mos_df = condition_add(
        mos_df, dtrpre_non_rmos, "Pre-DTX", "Right", "Non-Perturbation"
    )
    mos_df = condition_add(mos_df, dtrpre_per_lmos, "Pre-DTX", "Left", "Perturbation")
    mos_df = condition_add(mos_df, dtrpre_per_rmos, "Pre-DTX", "Right", "Perturbation")
    mos_df = condition_add(mos_df, dtrpre_sin_lmos, "Pre-DTX", "Left", "Sinusoidal")
    mos_df = condition_add(mos_df, dtrpre_sin_rmos, "Pre-DTX", "Right", "Sinusoidal")

    mos_df = condition_add(
        mos_df, dtrpost_non_lmos, "Post-DTX", "Left", "Non-Perturbation"
    )
    mos_df = condition_add(
        mos_df, dtrpost_non_rmos, "Post-DTX", "Right", "Non-Perturbation"
    )
    mos_df = condition_add(mos_df, dtrpost_per_lmos, "Post-DTX", "Left", "Perturbation")
    mos_df = condition_add(
        mos_df, dtrpost_per_rmos, "Post-DTX", "Right", "Perturbation"
    )
    mos_df = condition_add(mos_df, dtrpost_sin_lmos, "Post-DTX", "Left", "Sinusoidal")
    mos_df = condition_add(mos_df, dtrpost_sin_rmos, "Post-DTX", "Right", "Sinusoidal")

    # For just comparing between perturbation
    mos_combo = mos_df.drop(columns=["Limb"])
    con_mos_combo = mos_df.drop(columns=["Limb"])

    # Comparing conditions by perturbation state
    mos_non = mos_combo.loc[mos_combo["Perturbation_State"] == "Non-Perturbation"]
    mos_non = mos_non.drop(columns=["Perturbation_State"])
    mos_per = mos_combo.loc[mos_combo["Perturbation_State"] == "Perturbation"]
    mos_per = mos_per.drop(columns=["Perturbation_State"])
    mos_sin = mos_combo.loc[mos_combo["Perturbation_State"] == "Sinusoidal"]
    mos_sin = mos_sin.drop(columns=["Perturbation_State"])

    # Clean up dataframe for analysis as opposed to plotting
    mos_analysis_all = mos_combo.rename(
        columns={"Condition": "mouse_type", "Perturbation_State": "state", "MoS": "mos"}
    )
    mos_non_analysis = mos_non.rename(columns={"Condition": "mouse_type", "MoS": "mos"})
    mos_per_analysis = mos_per.rename(columns={"Condition": "mouse_type", "MoS": "mos"})
    mos_sin_analysis = mos_sin.rename(columns={"Condition": "mouse_type", "MoS": "mos"})

    # Stats model example
    # moore = sm.datasets.get_rdataset("Moore", "carData", cache=True)
    # data = moore.data
    #
    # data = data.rename(columns={"partner.status": "partner_status"})
    # print(data.head())
    #
    # moore_lm = ols()
    #
    # NOTE: Use Condition for partner_status/supp and Perturbation_State for fcategory/dose in comparison

    # H01: there is no significant effect in mos according to perturbation state
    # H02: there is no significant effect in mos according to the mouse strain/condition
    # H03: there are no interaction effects

    # Two-way ANOVA's comparing everything

    # Statsmodel method
    formula = "mos ~ C(mouse_type) + C(state) + C(mouse_type):C(state)"
    model = ols(formula, mos_analysis_all).fit()
    aov_table = anova_lm(model, type=2)
    print(f"Two way Anova done with statsmodels:\n{aov_table}\n")

    # Trying again with pingouin
    aov = pg.anova(
        dv="mos", between=["mouse_type", "state"], data=mos_analysis_all, detailed=True
    )

    print(f"Two way Anova done with pingouin:\n{aov}\n")

    # Post-hoc t-tests
    cmap = ["1", "#fb6a4a", "#08306b", "#4292c6", "#c6dbef"]
    heatmap_args = {
        "cmap": cmap,
        "linewidths": 0.25,
        "linecolor": "0.5",
        "clip_on": False,
        "square": True,
        "cbar_ax_bbox": [0.80, 0.35, 0.04, 0.3],
    }

    # Comparing aggregate of all state MoS values
    condition_post_hoc = sp.posthoc_conover(
        mos_analysis_all, val_col="mos", group_col="mouse_type", p_adjust="holm"
    )
    print(
        f"Post hoc comparison t-test comparing conditions including all states:\n{condition_post_hoc}\n"
    )

    # Comparing Non-Perturbed Data
    non_post_hoc = sp.posthoc_mannwhitney(
        mos_non_analysis, val_col="mos", group_col="mouse_type", p_adjust="holm"
    )
    print(
        f"Post hoc comparison t-test comparing conditions during unperturbed locomotion:\n{non_post_hoc}\n"
    )
    plt.title("Comparison of unperturbed locomotion")
    sp.sign_plot(non_post_hoc, **heatmap_args)
    # plt.savefig("./combined_figures/man-whitney/non_post_hoc_mannwhit.pdf", dpi=300)
    plt.show()

    # Comparing Cautious walking/Perturbed Data
    per_post_hoc = sp.posthoc_mannwhitney(
        mos_per_analysis, val_col="mos", group_col="mouse_type", p_adjust="holm"
    )
    print(
        f"Post hoc comparison t-test comparing conditions during randomly perturbed locomotion:\n{per_post_hoc}\n"
    )
    plt.title("Comparison of perturbed locomotion")
    sp.sign_plot(per_post_hoc, **heatmap_args)
    # plt.savefig("./combined_figures/man-whitney/per_post_hoc_mannwhit.pdf", dpi=300)
    plt.show()

    # Comparing Sinusoidal Stimulation Data
    sin_post_hoc = sp.posthoc_mannwhitney(
        mos_per_analysis, val_col="mos", group_col="mouse_type"
    )
    print(
        f"Post hoc comparison t-test comparing conditions during sinusoidal stimulation locomotion:\n{sin_post_hoc}\n"
    )
    plt.title("Comparison of sinusoidal stimulation locomotion")
    sp.sign_plot(sin_post_hoc, **heatmap_args)
    # plt.savefig("./combined_figures/man-whitney/sin_post_hoc_mannwhit.pdf")
    plt.show()

    # Comparing


if __name__ == "__main__":
    main()
