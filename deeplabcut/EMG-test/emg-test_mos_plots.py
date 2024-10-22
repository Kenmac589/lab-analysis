import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import scipy as sp
import seaborn as sns
from pandas import DataFrame as df

# from scipy import stats as st
from statannotations.Annotator import Annotator


def condition_add(input_df, file_list, condition, limb, print_neg=True):
    total_mos = np.array([])
    for i in range(len(file_list)):
        limb = limb
        mos_values = pd.read_csv(file_list[i], header=None)
        mos_values = mos_values.to_numpy()
        mos_values = mos_values.ravel()
        total_mos = np.concatenate((total_mos, mos_values), axis=None)
        for j in range(len(mos_values)):
            entry = mos_values[j]
            if print_neg is True:
                if entry < 0.0:
                    print(
                        f"File with negative detected: {file_list[i]} with value {entry}"
                    )

            mos_entry = [[condition, limb, entry]]

            input_df = input_df._append(
                pd.DataFrame(
                    mos_entry,
                    columns=["Condition", "Limb", "MoS (cm)"],
                ),
                ignore_index=True,
            )

    print(f"Descriptives for {condition} {limb} MoS")
    print(
        f"Forelimb mean: {np.mean(total_mos)} std: {np.std(total_mos)} n: {len(total_mos)}\n"
    )
    return input_df


def main():

    # Setting some info at beginning
    save_fig_and_df = False
    df_filename = "./emg-test_mos_values.csv"
    figure_title = "Effect of EMG Implantation on Margin of Stability"
    figure_filename = (
        "./emg-test-analysis/emg-test-figures/emg-test_mos_comparison-star-welchs.svg"
    )

    # Pre EMG Implantation

    m_one_premg_lmos = [
        "./emg-test-analysis/emg-test-data/emg-test-m1-pre-emg-lmos-03.csv",
        "./emg-test-analysis/emg-test-data/emg-test-m1-pre-emg-lmos-04.csv",
    ]

    m_one_premg_rmos = [
        "./emg-test-analysis/emg-test-data/emg-test-m1-pre-emg-rmos-03.csv",
        "./emg-test-analysis/emg-test-data/emg-test-m1-pre-emg-rmos-04.csv",
    ]

    m_two_premg_lmos = [
        "./emg-test-analysis/emg-test-data/emg-test-m2-pre-emg-lmos-00.csv",
        "./emg-test-analysis/emg-test-data/emg-test-m2-pre-emg-lmos-01.csv",
        "./emg-test-analysis/emg-test-data/emg-test-m2-pre-emg-lmos-02.csv",
        # "./emg-test-analysis/emg-test-data/emg-test-m2-pre-emg-lmos-03.csv",
    ]

    m_two_premg_rmos = [
        "./emg-test-analysis/emg-test-data/emg-test-m2-pre-emg-rmos-00.csv",
        "./emg-test-analysis/emg-test-data/emg-test-m2-pre-emg-rmos-01.csv",
        "./emg-test-analysis/emg-test-data/emg-test-m2-pre-emg-rmos-02.csv",
        # "./emg-test-analysis/emg-test-data/emg-test-m2-pre-emg-rmos-03.csv",
    ]

    # Post EMG Implantation

    # ./emg-test-analysis/emg-test-supercom-redo/emg-test-data/
    m_one_postemg_lmos = [
        "./emg-test-analysis/emg-test-supercom-redo/emg-test-data/emg-test-m1-post-emg-lmos-00.csv",
        "./emg-test-analysis/emg-test-supercom-redo/emg-test-data/emg-test-m1-post-emg-lmos-01.csv",
        "./emg-test-analysis/emg-test-supercom-redo/emg-test-data/emg-test-m1-post-emg-lmos-02.csv",
        "./emg-test-analysis/emg-test-supercom-redo/emg-test-data/emg-test-m1-post-emg-lmos-03.csv",
    ]

    m_one_postemg_rmos = [
        "./emg-test-analysis/emg-test-supercom-redo/emg-test-data/emg-test-m1-post-emg-rmos-00.csv",
        "./emg-test-analysis/emg-test-supercom-redo/emg-test-data/emg-test-m1-post-emg-rmos-01.csv",
        "./emg-test-analysis/emg-test-supercom-redo/emg-test-data/emg-test-m1-post-emg-rmos-02.csv",
        "./emg-test-analysis/emg-test-supercom-redo/emg-test-data/emg-test-m1-post-emg-rmos-03.csv",
    ]

    # NOTE: Vids 8 and 9 are good quality but on an unlevel treadmill
    m_two_postemg_lmos = [
        "./emg-test-analysis/emg-test-supercom-redo/emg-test-data/emg-test-m2-post-emg-lmos-01.csv",
        "./emg-test-analysis/emg-test-supercom-redo/emg-test-data/emg-test-m2-post-emg-lmos-08.csv",
        "./emg-test-analysis/emg-test-supercom-redo/emg-test-data/emg-test-m2-post-emg-lmos-09.csv",
    ]

    m_two_postemg_rmos = [
        "./emg-test-analysis/emg-test-supercom-redo/emg-test-data/emg-test-m2-post-emg-rmos-01.csv",
        "./emg-test-analysis/emg-test-supercom-redo/emg-test-data/emg-test-m2-post-emg-rmos-08.csv",
        "./emg-test-analysis/emg-test-supercom-redo/emg-test-data/emg-test-m2-post-emg-rmos-09.csv",
    ]

    # Initialize the Dataframe

    mos_df = df(columns=["Condition", "Limb", "MoS (cm)"])

    # Pre EMG Implantation
    mos_df = condition_add(mos_df, m_one_premg_lmos, "M1 Pre-EMG", "Left")
    mos_df = condition_add(mos_df, m_one_premg_rmos, "M1 Pre-EMG", "Right")
    mos_df = condition_add(mos_df, m_two_premg_lmos, "M2 Pre-EMG", "Left")
    mos_df = condition_add(mos_df, m_two_premg_rmos, "M2 Pre-EMG", "Right")

    # Post EMG Implantation
    mos_df = condition_add(mos_df, m_one_postemg_lmos, "M1 Post-EMG", "Left")
    mos_df = condition_add(mos_df, m_one_postemg_rmos, "M1 Post-EMG", "Right")
    mos_df = condition_add(mos_df, m_two_postemg_lmos, "M2 Post-EMG", "Left")
    mos_df = condition_add(mos_df, m_two_postemg_rmos, "M2 Post-EMG", "Right")

    # For just comparing between perturbation
    mos_combo = mos_df.drop(columns=["Limb"])
    con_mos_combo = mos_df.drop(columns=["Limb"])

    # Just First Mouse
    m_one_df = mos_df[
        (mos_df["Condition"] == "M1 Pre-EMG") | (mos_df["Condition"] == "M1 Post-EMG")
    ]
    m_one_combo = m_one_df.drop(columns=["Limb"])

    # Just Second Mouse
    m_two_df = mos_df[
        (mos_df["Condition"] == "M2 Pre-EMG") | (mos_df["Condition"] == "M2 Post-EMG")
    ]
    m_two_combo = m_two_df.drop(columns=["Limb"])

    # Comparing Pooled groups
    emg_comp = mos_combo
    emg_comp["Condition"] = emg_comp["Condition"].replace("M1 Pre-EMG", "Pre-EMG")
    emg_comp["Condition"] = emg_comp["Condition"].replace("M2 Pre-EMG", "Pre-EMG")
    emg_comp["Condition"] = emg_comp["Condition"].replace("M1 Post-EMG", "Post-EMG")
    emg_comp["Condition"] = emg_comp["Condition"].replace("M2 Post-EMG", "Post-EMG")

    pool_pairs = [("Pre-EMG", "Post-EMG")]

    pre_emg_ind = emg_comp[emg_comp["Condition"] == "Pre-EMG"].index.values
    pre_emg_np = np.array(emg_comp.loc[pre_emg_ind, "MoS (cm)"])

    post_emg_ind = emg_comp[emg_comp["Condition"] == "Post-EMG"].index.values
    post_emg_np = np.array(emg_comp.loc[post_emg_ind, "MoS (cm)"])

    if len(pre_emg_np) != len(post_emg_np):
        if len(pre_emg_np) > len(post_emg_np):
            post_emg_np = np.pad(
                post_emg_np,
                (0, len(pre_emg_np) - len(post_emg_np)),
                "constant",
                constant_values=np.nan,
            )
        else:
            pre_emg_np = np.pad(
                pre_emg_np,
                (0, len(post_emg_np) - len(pre_emg_np)),
                "constant",
                constant_values=np.nan,
            )

    # Paired t-test
    tpair_results = sp.stats.ttest_rel(pre_emg_np, post_emg_np, nan_policy="omit")

    # Independent test
    tind_results = sp.stats.ttest_ind(pre_emg_np, post_emg_np, nan_policy="omit")

    # Pingouin based test
    tpg_res = pg.ttest(pre_emg_np, post_emg_ind, paired=True, confidence=0.95)

    # print(pre_emg_np)
    # print(f"Pre-EMG SD: {np.std(pre_emg_np)}\n")
    # print(f"Post-EMG SD: {np.std(post_emg_np)}\n")

    # Plotting
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set(style="white", font_scale=1.6, palette="colorblind", rc=custom_params)

    limb_order = ["Left", "Right"]

    # Ordering for the plots
    one_order = ["M1 Pre-EMG", "M1 Post-EMG"]
    one_pairs = [
        # [("M1 Pre-EMG", "Left"), ("M1 Pre-EMG", "Right")],
        # [("M1 Post-EMG", "Left"), ("M1 Post-EMG", "Right")],
        # [("M1 Pre-EMG", "Left"), ("M1 Post-EMG", "Left")],
        # [("M1 Pre-EMG", "Right"), ("M1 Post-EMG", "Right")],
        # [("Left", "M1 Pre-EMG"), ("Right", "M1 Pre-EMG")],
        # [("Left", "M1 Post-EMG"), ("Right", "M1 Post-EMG")],
        [("Left", "M1 Pre-EMG"), ("Left", "M1 Post-EMG")],
        [("Right", "M1 Pre-EMG"), ("Right", "M1 Post-EMG")],
    ]
    two_order = ["M2 Pre-EMG", "M2 Post-EMG"]
    two_pairs = [
        # [("M2 Pre-EMG", "Left"), ("M2 Pre-EMG", "Right")],
        # [("M2 Post-EMG", "Left"), ("M2 Post-EMG", "Right")],
        # [("M2 Pre-EMG", "Left"), ("M2 Post-EMG", "Left")],
        # [("M2 Pre-EMG", "Right"), ("M2 Post-EMG", "Right")],
        # [("Left", "M2 Pre-EMG"), ("Right", "M2 Pre-EMG")],
        # [("Left", "M2 Post-EMG"), ("Right", "M2 Post-EMG")],
        [("Left", "M2 Pre-EMG"), ("Left", "M2 Post-EMG")],
        [("Right", "M2 Pre-EMG"), ("Right", "M2 Post-EMG")],
    ]

    m_one_param = {
        "data": m_one_df,
        "x": "Limb",
        "y": "MoS (cm)",
        "hue": "Condition",
        "hue_order": one_order,
    }

    m_two_param = {
        "data": m_two_df,
        "x": "Limb",
        "y": "MoS (cm)",
        "hue": "Condition",
        "hue_order": two_order,
    }

    # Individual Limb Analysis
    fig = plt.figure(figsize=(15.8, 10.80))
    axs = fig.subplot_mosaic(
        [
            ["m_two", "m_one"],
            ["m_two", "m_one"],
        ]
    )

    # Mouse 1 Figure
    m_one_plot = sns.barplot(**m_one_param, ci=95, capsize=0.05, ax=axs["m_one"])
    axs["m_one"].set_title("MoS for M1 EMG-test")
    axs["m_one"].legend(loc="upper left", fontsize=14)
    annotator = Annotator(m_one_plot, one_pairs, **m_one_param)
    annotator.new_plot(m_one_plot, one_pairs, plot="barplot", **m_one_param)
    annotator.configure(
        hide_non_significant=False,
        test="t-test_welch",
        text_format="star",
        loc="inside",
    )
    annotator.apply_test().annotate(line_offset_to_group=0.2, line_offset=0.1)

    # Mouse 2 Figure
    m_two_plot = sns.barplot(**m_two_param, ci=95, capsize=0.05, ax=axs["m_two"])
    axs["m_two"].set_title("MoS for M2 EMG-test")
    axs["m_one"].set_title("MoS for M1 EMG-test")
    axs["m_two"].legend(loc="upper left", fontsize=14)
    annotator = Annotator(m_two_plot, two_pairs, **m_two_param)
    annotator.new_plot(m_two_plot, two_pairs, plot="barplot", **m_two_param)
    annotator.configure(
        hide_non_significant=False,
        test="t-test_welch",
        text_format="star",
        loc="inside",
    )
    annotator.apply_test().annotate(line_offset_to_group=0.2, line_offset=0.1)

    # Displaying Figure
    fig = plt.gcf()
    fig.set_size_inches(15.8, 10.80)
    fig.tight_layout()

    if save_fig_and_df is True:
        plt.savefig(figure_filename, dpi=300)

    # plt.show()


if __name__ == "__main__":
    main()
