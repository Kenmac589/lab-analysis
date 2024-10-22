import dlc2kinematics as dlck
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import scipy as sp
import seaborn as sns
from kinsynpy import dlctools as dlt
from pandas import DataFrame as df
# from scipy import stats as st
from statannotations.Annotator import Annotator


def dlc_sw(
    h5_file,
    calib_markers=[
        "calib_1",
        "calib_2",
        "calib_3",
        "calib_4",
        "calib_5",
        "calib_6",
    ],
):
    """Calculates and returns step width from h5 file

    Parameters
    ----------
    hf_file: "str"
        File path to the h5 file you want to analyze

    Returns
    -------
    fl_sw: numpy.ndarray
        1-D array of step widths calculated for the forelimb
    hl_sw: numpy.ndarray
        1-D array of step widths calculated for the hindlimb

    """

    df, bodyparts, scorer = dlck.load_data(h5_file)
    calib_factor = dlt.dlc_calibrate(df, bodyparts, scorer, calib_markers)

    rfly_np = dlt.mark_process(df, scorer, "mirror_rfl", "y", calib_factor)
    rhly_np = dlt.mark_process(df, scorer, "mirror_rhl", "y", calib_factor)
    lfly_np = dlt.mark_process(df, scorer, "mirror_lfl", "y", calib_factor)
    lhly_np = dlt.mark_process(df, scorer, "mirror_lhl", "y", calib_factor)
    rflx_np = dlt.mark_process(df, scorer, "mirror_rfl", "x", calib_factor)
    rhlx_np = dlt.mark_process(df, scorer, "mirror_rhl", "x", calib_factor)
    lflx_np = dlt.mark_process(df, scorer, "mirror_lfl", "x", calib_factor)
    lhlx_np = dlt.mark_process(df, scorer, "mirror_lhl", "x", calib_factor)

    fl_sw = dlt.step_width_est(rl_x=rflx_np, ll_x=lflx_np, rl_y=rfly_np, ll_y=lfly_np)
    hl_sw = dlt.step_width_est(rl_x=rhlx_np, ll_x=lhlx_np, rl_y=rhly_np, ll_y=lhly_np)

    return fl_sw, hl_sw


def dlc_hiph(
    h5_file,
    calib_markers=[
        "calib_1",
        "calib_2",
        "calib_3",
        "calib_4",
        "calib_5",
        "calib_6",
    ],
):
    """Calculates and returns step width from h5 file

    Parameters
    ----------
    hf_file: "str"
        File path to the h5 file you want to analyze

    Returns
    -------
    hip_h:
        Hip height in (cm)

    """

    df, bodyparts, scorer = dlck.load_data(h5_file)
    calib_factor = dlt.dlc_calibrate(df, bodyparts, scorer, calib_markers)

    toey_np = dlt.mark_process(df, scorer, "toe", "y", calib_factor)
    hipy_np = dlt.mark_process(df, scorer, "hip", "y", calib_factor)

    hip_h = dlt.hip_height(toey_np, hipy_np)

    return hip_h


def dlc_sw_cond_add(input_df, h5_path_list, condition):
    total_fl_sw = np.array([])
    total_hl_sw = np.array([])

    for i in range(len(h5_path_list)):
        fl_sw, hl_sw = dlc_sw(h5_path_list[i])
        total_fl_sw = np.concatenate((total_fl_sw, fl_sw), axis=None)
        total_hl_sw = np.concatenate((total_hl_sw, hl_sw), axis=None)

        # Adding forelimb
        limb = "Forelimb"
        for j in range(len(fl_sw)):
            entry = fl_sw[j]
            step_width_entry = [[condition, limb, entry]]
            input_df = input_df._append(
                pd.DataFrame(
                    step_width_entry,
                    columns=[
                        "Condition",
                        "Limb",
                        "Step Width (cm)",
                    ],
                ),
                ignore_index=True,
            )

        limb = "Hindlimb"
        for j in range(len(hl_sw)):
            entry = hl_sw[j]
            step_width_entry = [[condition, limb, entry]]
            input_df = input_df._append(
                pd.DataFrame(
                    step_width_entry,
                    columns=[
                        "Condition",
                        "Limb",
                        "Step Width (cm)",
                    ],
                ),
                ignore_index=True,
            )

    print(f"Descriptives for {condition}")
    print(
        f"Forelimb mean: {np.mean(total_fl_sw)} std: {np.std(total_fl_sw)} n: {len(total_hl_sw)}"
    )
    print(
        f"Hindlimb mean: {np.mean(total_hl_sw)} std: {np.std(total_hl_sw)} n: {len(total_fl_sw)}\n"
    )

    return input_df


def dlc_hiph_cond_add(input_df, h5_path_list, condition, perturbation_state):
    for i in range(len(h5_path_list)):
        hip_height = dlc_hiph(h5_path_list[i])
        entry = hip_height
        print(hip_height)

        hiph_entry = [[condition, perturbation_state, entry]]

        input_df = input_df._append(
            pd.DataFrame(
                hiph_entry,
                columns=["Condition", "Perturbation State", "Hip Height (cm)"],
            ),
            ignore_index=True,
        )

    return input_df


def condition_add_adj_dlc(
    input_df, file_list, h5_file, condition, limb, perturbation_state, print_neg=False
):
    for i in range(len(file_list)):
        limb = limb
        perturbation_state = perturbation_state
        mos_values = pd.read_csv(file_list[i], header=None)
        fl_sw, hl_sw = dlc_sw(h5_file=h5_file[i])
        mos_values = mos_values.to_numpy()
        mos_values = mos_values.ravel()
        mos_values = dlt.stepw_mos_corr(
            fl_stepw=fl_sw, hl_stepw=hl_sw, mos_values=mos_values
        )
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
                    columns=["Condition", "Limb", "MoS (cm)"],
                ),
                ignore_index=True,
            )

    return input_df


def condition_add(input_df, file_list, condition, limb, print_neg=True):
    for i in range(len(file_list)):
        limb = limb
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

            mos_entry = [[condition, limb, entry]]

            input_df = input_df._append(
                pd.DataFrame(
                    mos_entry,
                    columns=["Condition", "Limb", "MoS (cm)"],
                ),
                ignore_index=True,
            )

    return input_df


def main():

    # Setting some info at beginning
    save_fig_and_df = False
    # df_filename = "./emg-test_mos_values.csv"
    figure_title = "Effect of EMG Implantation on Step Width"
    figure_filename = "./emg-test-analysis/emg-test-figures/emg-test_stepw.pdf"

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

    # FOR STEP WIDTH

    m_one_preemg = [
        "./emg-test_redo/EMG-test-1-pre-emg_000003DLC_resnet50_supercomAug13shuffle1_1000000_filtered.h5",
        "./emg-test_redo/EMG-test-1-pre-emg_000004DLC_resnet50_supercomAug13shuffle1_1000000_filtered.h5",
    ]
    m_two_preemg = [
        "./emg-test_redo/EMG-test-2-pre-emg_000000DLC_resnet50_supercomAug13shuffle1_1000000_filtered.h5",
        "./emg-test_redo/EMG-test-2-pre-emg_000001DLC_resnet50_supercomAug13shuffle1_1000000_filtered.h5",
        "./emg-test_redo/EMG-test-2-pre-emg_000002DLC_resnet50_supercomAug13shuffle1_1000000_filtered.h5",
    ]

    m_one_postemg = [
        "./emg-test_redo/EMG-test-1-post-emg_000000DLC_resnet50_supercomAug13shuffle1_1000000_filtered.h5",
        "./emg-test_redo/EMG-test-1-post-emg_000001DLC_resnet50_supercomAug13shuffle1_1000000_filtered.h5",
        "./emg-test_redo/EMG-test-1-post-emg_000002DLC_resnet50_supercomAug13shuffle1_1000000_filtered.h5",
        "./emg-test_redo/EMG-test-1-post-emg_000003DLC_resnet50_supercomAug13shuffle1_1000000_filtered.h5",
    ]

    m_two_postemg = [
        "./emg-test_redo/EMG-test-2-post-emg-2024-07-24_000000DLC_resnet50_supercomAug13shuffle1_1000000_filtered.h5",
        "./emg-test_redo/EMG-test-2-post-emg-2024-07-24_000008DLC_resnet50_supercomAug13shuffle1_1000000_filtered.h5",
        "./emg-test_redo/EMG-test-2-post-emg-2024-07-24_000009DLC_resnet50_supercomAug13shuffle1_1000000_filtered.h5",
    ]

    # Initialize the Dataframe

    mos_df = df(columns=["Condition", "Limb", "MoS (cm)"])
    step_width_df = df(columns=["Condition", "Limb", "Step Width (cm)"])

    # For MoS

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

    # For Step Width
    step_width_df = dlc_sw_cond_add(step_width_df, m_one_preemg, "M1 Pre-EMG")
    step_width_df = dlc_sw_cond_add(step_width_df, m_two_preemg, "M2 Pre-EMG")
    step_width_df = dlc_sw_cond_add(step_width_df, m_one_postemg, "M1 Post-EMG")
    step_width_df = dlc_sw_cond_add(step_width_df, m_two_postemg, "M2 Post-EMG")

    # For just comparing between perturbation
    mos_combo = mos_df.drop(columns=["Limb"])
    con_mos_combo = mos_df.drop(columns=["Limb"])

    # step_width_df.describe()

    # Just First Mouse
    m_one_df = step_width_df[
        (step_width_df["Condition"] == "M1 Pre-EMG")
        | (step_width_df["Condition"] == "M1 Post-EMG")
    ]
    m_one_combo = m_one_df.drop(columns=["Limb"])

    # Just Second Mouse
    m_two_df = step_width_df[
        (step_width_df["Condition"] == "M2 Pre-EMG")
        | (step_width_df["Condition"] == "M2 Post-EMG")
    ]
    m_two_combo = m_two_df.drop(columns=["Limb"])

    # m_two_postfl = m_two_df[
    #     (m_two_df["Condition"] == "M2 Post-EMG") & (m_two_df["Limb"] == "Forelimb")
    # ]
    # print(m_two_postfl.describe())

    # Comparing Pooled groups
    emg_comp = mos_combo
    emg_comp["Condition"] = emg_comp["Condition"].replace("M1 Pre-EMG", "Pre-EMG")
    emg_comp["Condition"] = emg_comp["Condition"].replace("M2 Pre-EMG", "Pre-EMG")
    emg_comp["Condition"] = emg_comp["Condition"].replace("M1 Post-EMG", "Post-EMG")
    emg_comp["Condition"] = emg_comp["Condition"].replace("M2 Post-EMG", "Post-EMG")

    # Plotting
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set(style="white", font_scale=1.6, palette="colorblind", rc=custom_params)

    limb_order = ["Forelimb", "Hindlimb"]

    # Ordering for the plots
    one_order = ["M1 Pre-EMG", "M1 Post-EMG"]
    one_pairs = [
        # [("M1 Pre-EMG", "Forelimb"), ("M1 Pre-EMG", "Hindlimb")],
        # [("M1 Post-EMG", "Forelimb"), ("M1 Post-EMG", "Hindlimb")],
        # [("M1 Pre-EMG", "Forelimb"), ("M1 Post-EMG", "Forelimb")],
        # [("M1 Pre-EMG", "Hindlimb"), ("M1 Post-EMG", "Hindlimb")],
        # [("Forelimb", "M1 Pre-EMG"), ("Hindlimb", "M1 Pre-EMG")],
        # [("Forelimb", "M1 Post-EMG"), ("Hindlimb", "M1 Post-EMG")],
        [("Forelimb", "M1 Pre-EMG"), ("Forelimb", "M1 Post-EMG")],
        [("Hindlimb", "M1 Pre-EMG"), ("Hindlimb", "M1 Post-EMG")],
    ]
    two_order = ["M2 Pre-EMG", "M2 Post-EMG"]
    two_pairs = [
        # [("M2 Pre-EMG", "Forelimb"), ("M2 Pre-EMG", "Hindlimb")],
        # [("M2 Post-EMG", "Forelimb"), ("M2 Post-EMG", "Hindlimb")],
        # [("M2 Pre-EMG", "Forelimb"), ("M2 Post-EMG", "Forelimb")],
        # [("M2 Pre-EMG", "Hindlimb"), ("M2 Post-EMG", "Hindlimb")],
        # [("Forelimb", "M2 Pre-EMG"), ("Hindlimb", "M2 Pre-EMG")],
        # [("Forelimb", "M2 Post-EMG"), ("Hindlimb", "M2 Post-EMG")],
        [("Forelimb", "M2 Pre-EMG"), ("Forelimb", "M2 Post-EMG")],
        [("Hindlimb", "M2 Pre-EMG"), ("Hindlimb", "M2 Post-EMG")],
    ]

    m_one_param = {
        "data": m_one_df,
        "x": "Limb",
        "y": "Step Width (cm)",
        "hue": "Condition",
        "hue_order": one_order,
    }

    m_two_param = {
        "data": m_two_df,
        "x": "Limb",
        "y": "Step Width (cm)",
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
    axs["m_one"].set_title("Step Width for M1 EMG-test")
    axs["m_one"].legend(loc="upper left", fontsize=14)
    annotator = Annotator(m_one_plot, one_pairs, **m_one_param)
    annotator.new_plot(m_one_plot, one_pairs, plot="barplot", **m_one_param)
    annotator.configure(
        hide_non_significant=False,
        test="t-test_welch",
        text_format="simple",
        loc="inside",
    )
    annotator.apply_test().annotate(line_offset_to_group=0.2, line_offset=0.1)

    # Mouse 2 Figure
    m_two_plot = sns.barplot(**m_two_param, ci=95, capsize=0.05, ax=axs["m_two"])
    axs["m_two"].set_title("Step Width for M2 EMG-test")
    axs["m_two"].legend(loc="upper left", fontsize=14)
    annotator = Annotator(m_two_plot, two_pairs, **m_two_param)
    annotator.new_plot(m_two_plot, two_pairs, plot="barplot", **m_two_param)
    annotator.configure(
        hide_non_significant=False,
        test="t-test_welch",
        text_format="simple",
        loc="inside",
    )
    annotator.apply_test().annotate(line_offset_to_group=0.2, line_offset=0.1)

    # Displaying Figure
    fig = plt.gcf()
    fig.set_size_inches(15.8, 10.80)
    fig.tight_layout()

    if save_fig_and_df is True:
        plt.savefig(figure_filename, dpi=300)

    plt.show()


if __name__ == "__main__":
    main()
