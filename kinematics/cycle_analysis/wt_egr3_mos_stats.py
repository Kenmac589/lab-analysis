import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import statsmodels.api as sm
from pandas import DataFrame as df
from scipy import stats
from statannotations.Annotator import Annotator
from statsmodels.formula.api import ols
from statsmodels.graphics.factorplots import interaction_plot
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

    # For just comparing between perturbation
    mos_combo = mos_df.drop(columns=["Limb"])
    con_mos_combo = mos_df.drop(columns=["Limb"])

    # Clean up dataframe for analysis as opposed to plotting
    mos_analysis = mos_combo.rename(
        columns={"Condition": "mouse_type", "Perturbation_State": "state", "MoS": "mos"}
    )

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

    # Statsmodel method
    formula = "mos ~ C(mouse_type) + C(state) + C(mouse_type):C(state)"
    model = ols(formula, mos_analysis).fit()
    aov_table = anova_lm(model, type=2)

    print("Two way Anova done with statsmodels")
    print(aov_table)
    print()

    # Trying again with pingouin
    aov = pg.anova(
        dv="mos", between=["mouse_type", "state"], data=mos_analysis, detailed=True
    )

    print("Two way Anova done with pingouin")
    print(aov)


if __name__ == "__main__":
    main()
