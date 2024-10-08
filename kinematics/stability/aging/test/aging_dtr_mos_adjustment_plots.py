# from scipy import stats as st
import dlc2kinematics as dlck
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from kinsynpy import dlctools as dlt
from kinsynpy import latstability as ls
from pandas import DataFrame as df
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


def dlc_sw_cond_add(input_df, h5_path_list, condition, limb, perturbation_state):
    for i in range(len(h5_path_list)):
        fl_sw, hl_sw = dlc_sw(h5_path_list[i])

        if limb == "Forelimb":
            entries = fl_sw
        elif limb == "Hindlimb":
            entries = hl_sw
        else:
            print("Check limb label")

        for j in range(len(entries)):
            entry = entries[j]
            step_width_entry = [[condition, limb, perturbation_state, entry]]
            input_df = input_df._append(
                pd.DataFrame(
                    step_width_entry,
                    columns=[
                        "Condition",
                        "Limb",
                        "Perturbation State",
                        "Step Width (cm)",
                    ],
                ),
                ignore_index=True,
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


def step_width_batch(inputdf, event_channels, y_channels):
    """Doing step width calculation in one go
    :param inputdf: A spike file input as *.csv or formatted as such
    :param event_channels: A list with all the proper channel names for the event channels
    :param y_channels: A list with all the proper channel names for the channels from DLC
    :note: The proper order for event channels goes lhl, lfl, rhl, rfl with swonset first.

    :return fl_step_widths: array of step width values for the forelimb
    :return hl_step_widths: array of step width values for the hindlimb
    """

    # For forelimb
    fl_step_widths = ls.step_width(
        inputdf,
        rl_swoff=event_channels[7],
        ll_swoff=event_channels[3],
        rl_y="35 FRy (cm)",
        ll_y="33 FLy (cm)",
    )
    hl_step_widths = ls.step_width(
        inputdf,
        rl_swoff=event_channels[5],
        ll_swoff=event_channels[1],
        rl_y="30 HRy (cm)",
        ll_y="28 HLy (cm)",
    )

    return fl_step_widths, hl_step_widths


def step_width_batch_est(inputdf, x_channels, y_channels):
    """Doing step width calculation in one go
    :param inputdf: A spike file input as *.csv or formatted as such
    :param event_channels: A list with all the proper channel names for the event channels
    :param y_channels: A list with all the proper channel names for the channels from DLC
    :note: The proper order for event channels goes lhl, lfl, rhl, rfl with swonset first.

    :return fl_step_widths: array of step width values for the forelimb
    :return hl_step_widths: array of step width values for the hindlimb
    """

    # For forelimb
    fl_step_widths = ls.step_width_est(
        inputdf,
        rl_x=x_channels[0],
        ll_x=x_channels[1],
        rl_y=y_channels[0],
        ll_y=y_channels[1],
    )
    hl_step_widths = ls.step_width_est(
        inputdf,
        rl_x=x_channels[2],
        ll_x=x_channels[3],
        rl_y=y_channels[2],
        ll_y=y_channels[3],
    )

    return fl_step_widths, hl_step_widths


def sw_condition_add(
    input_df, file_list, x_chan, y_chan, condition, limb, perturbation_state
):
    for i in range(len(file_list)):
        current_file = pd.read_csv(file_list[i])
        step_width_values = ls.step_width_est(
            current_file,
            rl_x=x_chan[0],
            ll_x=x_chan[1],
            rl_y=y_chan[0],
            ll_y=y_chan[1],
        )
        limb = limb
        perturbation_state = perturbation_state
        step_width_values = np.array(step_width_values, dtype=float)
        # print(len(step_width_values))
        step_width_values = step_width_values.ravel()

        for j in range(len(step_width_values)):
            entry = step_width_values[j]
            step_width_entry = [[condition, limb, perturbation_state, entry]]

            input_df = input_df._append(
                pd.DataFrame(
                    step_width_entry,
                    columns=[
                        "Condition",
                        "Limb",
                        "Perturbation State",
                        "Step Width (cm)",
                    ],
                ),
                ignore_index=True,
            )

    return input_df


def hiph_condition_add(
    input_df, file_list, hhy_channels, condition, perturbation_state
):
    for i in range(len(file_list)):
        current_file = pd.read_csv(file_list[i])
        hip_height = ls.hip_height(
            current_file,
            toey=hhy_channels[0],
            hipy=hhy_channels[1],
        )
        perturbation_state = perturbation_state
        entry = hip_height

        hiph_entry = [[condition, perturbation_state, entry]]

        input_df = input_df._append(
            pd.DataFrame(
                hiph_entry,
                columns=["Condition", "Perturbation State", "Hip Height (cm)"],
            ),
            ignore_index=True,
        )

    return input_df


def condition_add(
    input_df, file_list, condition, limb, perturbation_state, print_neg=False
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


def condition_add_adj(
    input_df,
    mos_file_test,
    file_list,
    x_channel_list,
    y_channel_list,
    condition,
    limb,
    perturbation_state,
    print_neg=False,
):

    fl_x = x_channel_list[0:2]
    hl_x = x_channel_list[2:4]
    fl_y = y_channel_list[0:2]
    hl_y = y_channel_list[2:4]

    for i in range(len(mos_file_test)):
        limb = limb
        perturbation_state = perturbation_state
        full_file = pd.read_csv(file_list[i], header=0)
        mos_values = pd.read_csv(mos_file_test[i], header=None)
        fl_sw = ls.step_width_est(
            full_file, rl_x=fl_x[0], ll_x=fl_x[1], rl_y=fl_y[0], ll_y=fl_y[1]
        )
        hl_sw = ls.step_width_est(
            full_file, rl_x=hl_x[0], ll_x=hl_x[1], rl_y=hl_y[0], ll_y=hl_y[1]
        )
        mos_values = mos_values.to_numpy()
        mos_values = mos_values.ravel()
        mos_values = dlt.stepw_mos_corr(
            fl_stepw=fl_sw, hl_stepw=hl_sw, mos_values=mos_values
        )

        # print(f"Average MoS for file {mos_file_test[i]}: {np.mean(mos_values)}")
        for j in range(len(mos_values)):
            entry = mos_values[j]
            if print_neg is True:
                if entry < 0.0:
                    print(
                        f"File with negative detected: {mos_file_test[i]} with value {entry}"
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
                    columns=["Condition", "Limb", "Perturbation State", "MoS (cm)"],
                ),
                ignore_index=True,
            )

    return input_df


def main():

    # Setting some info at beginning
    save_fig_and_df = False
    df_filename = "./aging_mos_4m_12m_18m_dtr-adjusted.csv"
    figure_title = "MoS adjusted for Step Width between 4, 12 and 18 month old DTX Mice Pre and Post Injection"
    figure_filename = "./aging/aging_for_grant/aging_mos_all_groups-adjusted-violin.svg"

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
        # "./dtr_data/postdtx/postdtx_5per_lmos-2.csv",
        "./dtr_data/postdtx/postdtx_6per_lmos-auto.csv",
    ]

    dtrpost_per_rmos = [
        "./dtr_data/postdtx/postdtx_2per_rmos.csv",
        "./dtr_data/postdtx/postdtx_3per_rmos.csv",
        "./dtr_data/postdtx/postdtx_5per_rmos-1.csv",
        # "./dtr_data/postdtx/postdtx_5per_rmos-2.csv",
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

    # Data files for step widths

    # NOTE: WT channel names also work for Egr3
    wt_y_channels = ["35 FRy (cm)", "33 FLy (cm)", "30 HRy (cm)", "28 HLy (cm)"]
    wt_x_channels = ["34 FRx (cm)", "32 FLx (cm)", "29 HRx (cm)", "27 HLx (cm)"]

    # wt_fl_x_channels = ["34 FRx (cm)", "32 FLx (cm)"]
    # wt_fl_y_channels = ["35 FRy (cm)", "33 FLy (cm)"]
    # wt_hl_x_channels = ["29 HRx (cm)", "27 HLx (cm)"]
    # wt_hl_y_channels = ["30 HRy (cm)", "28 HLy (cm)"]

    # wt_hh_channels = ["24 toey (cm)", "16 Hipy (cm)"]

    dtr_x_channels = ["39 FRx (cm)", "37 FLx (cm)", "35 HRx (cm)", "33 HLx (cm)"]
    dtr_y_channels = ["40 FRy (cm)", "38 FLy (cm)", "36 HRy (cm)", "34 HLy (cm)"]
    # dtr_fl_x_channels = ["39 FRx (cm)", "37 FLx (cm)"]
    # dtr_fl_y_channels = ["40 FRy (cm)", "38 FLy (cm)"]
    # dtr_hl_x_channels = ["35 HRx (cm)", "33 HLx (cm)"]
    # dtr_hl_y_channels = ["36 HRy (cm)", "34 HLy (cm)"]

    # dtr_hh_channels = ["25 toey (cm)", "17 Hipy (cm)"]

    wt_non = [
        "./wt_data/wt-1-non-all.txt",
        "./wt_data/wt-2-non-all.txt",
        "./wt_data/wt-3-non-all.txt",
        "./wt_data/wt-4-non-all.txt",
        "./wt_data/wt-5-non-all.txt",
    ]
    wt_per = [
        "./wt_data/wt-1-per-all.txt",
        "./wt_data/wt-2-per-all.txt",
        "./wt_data/wt-3-per-all.txt",
        "./wt_data/wt-4-per-all.txt",
        "./wt_data/wt-5-per-all.txt",
    ]
    wt_sin = [
        "./wt_data/wt-1-sin-all.txt",
        "./wt_data/wt-2-sin-all.txt",
        "./wt_data/wt-3-sin-all.txt",
        "./wt_data/wt-4-sin-all.txt",
    ]

    egr3_non = [
        "./egr3_data/egr3-6-non-all.txt",
        "./egr3_data/egr3-7-non-all.txt",
        "./egr3_data/egr3-8-non-all.txt",
        "./egr3_data/egr3-9-non-all.txt",
        "./egr3_data/egr3-10-non-all.txt",
    ]
    egr3_per = [
        "./egr3_data/egr3-6-per-all.txt",
        "./egr3_data/egr3-7-per-all.txt",
        "./egr3_data/egr3-8-per-all.txt",
        "./egr3_data/egr3-9-per-all-1.txt",
        "./egr3_data/egr3-9-per-all-2.txt",
        "./egr3_data/egr3-10-per-all-1.txt",
        "./egr3_data/egr3-10-per-all-2.txt",
    ]
    egr3_sin = [
        "./egr3_data/egr3-6-sin-all.txt",
        "./egr3_data/egr3-7-sin-all.txt",
        "./egr3_data/egr3-8-sin-all.txt",
        "./egr3_data/egr3-9-sin-all-1.txt",
        "./egr3_data/egr3-9-sin-all-2.txt",
        "./egr3_data/egr3-10-sin-all.txt",
    ]

    dtrpre_non = [
        "./dtr_data/predtx/dtr-pre-2-non-all.txt",
        "./dtr_data/predtx/dtr-pre-3-non-all.txt",
        "./dtr_data/predtx/dtr-pre-5-non-all.txt",
        "./dtr_data/predtx/dtr-pre-6-non-all.txt",
        "./dtr_data/predtx/dtr-pre-7-non-all.txt",
    ]
    dtrpre_per = [
        "./dtr_data/predtx/dtr-pre-2-per-all.txt",
        "./dtr_data/predtx/dtr-pre-3-per-all.txt",
        "./dtr_data/predtx/dtr-pre-5-per-all-1.txt",
        "./dtr_data/predtx/dtr-pre-5-per-all-2.txt",
        "./dtr_data/predtx/dtr-pre-6-per-all.txt",
        "./dtr_data/predtx/dtr-pre-7-per-all.txt",
    ]
    dtrpre_sin = [
        "./dtr_data/predtx/dtr-pre-2-sin-all.txt",
        # "./dtr_data/predtx/dtr-pre-3-sin-all.txt",
        "./dtr_data/predtx/dtr-pre-3-sin-all-1.txt",
        "./dtr_data/predtx/dtr-pre-3-sin-all-2.txt",
        "./dtr_data/predtx/dtr-pre-5-sin-all.txt",
        "./dtr_data/predtx/dtr-pre-6-sin-all.txt",
        "./dtr_data/predtx/dtr-pre-7-sin-all.txt",
    ]

    dtrpost_non = [
        "./dtr_data/postdtx/dtr-post-2-non-all.txt",
        "./dtr_data/postdtx/dtr-post-3-non-all.txt",
        "./dtr_data/postdtx/dtr-post-5-non-all.txt",
        "./dtr_data/postdtx/dtr-post-6-non-all.txt",
        # "./dtr_data/postdtx/dtr-post-8-non-all.txt",
    ]
    dtrpost_per = [
        "./dtr_data/postdtx/dtr-post-2-per-all.txt",
        "./dtr_data/postdtx/dtr-post-3-per-all.txt",
        "./dtr_data/postdtx/dtr-post-5-per-all.txt",
        # "./dtr_data/postdtx/dtr-post-5-per-xcom-1.txt",
        "./dtr_data/postdtx/dtr-post-6-per-all.txt",
    ]
    dtrpost_sin = [
        "./dtr_data/postdtx/dtr-post-2-sin-all.txt",
        # "./dtr_data/postdtx/dtr-post-3-sin-all.txt",
        "./dtr_data/postdtx/dtr-post-5-sin-all.txt",
        "./dtr_data/postdtx/dtr-post-6-sin-all.txt",
    ]

    # 12m Mice MoS Values
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

    # 12m Post-DTX MoS Values
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

    # 18m Mouse MoS Values
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

    # Data files for step widths in old mice

    # Adding 12m Aged Mice
    age_dtrpre_non = [
        "./aging/12mo/1yr-dtr_norosa-1/1yr-dtr_norosa-1-predtx/1yrDTRnoRosa-M1-19102023_000000DLC_resnet50_1yrDTRnoRosa-preDTXJan31shuffle1_1030000_filtered.h5",
        "./aging/12mo/1yr-dtr_norosa-2/1yr-dtr_norosa-2-predtx/1yrDTRnoRosa-M2-preDTX-23102023_000000DLC_resnet50_1yrDTRnoRosa-preDTXJan31shuffle1_1030000_filtered.h5",
        # "./aging/12mo/1yr-dtr_norosa-2/1yr-dtr_norosa-2-predtx/1yrDTRnoRosa-M2-preDTX-23102023_000001DLC_resnet50_1yrDTRnoRosa-preDTXJan31shuffle1_1030000_filtered.h5",
    ]
    age_dtrpre_per = [
        "./aging/12mo/1yr-dtr_norosa-1/1yr-dtr_norosa-1-predtx/1yrDTRnoRosa-M1-19102023_000009DLC_resnet50_1yrDTRnoRosa-preDTXJan31shuffle1_1030000_filtered.h5",
        # "./aging/12mo/1yr-dtr_norosa-1/1yr-dtr_norosa-1-predtx/1yrDTRnoRosa-M1-19102023_000010DLC_resnet50_1yrDTRnoRosa-preDTXJan31shuffle1_1030000_filtered.h5",
        "./aging/12mo/1yr-dtr_norosa-2/1yr-dtr_norosa-2-predtx/1yrDTRnoRosa-M2-preDTX-23102023_000016DLC_resnet50_1yrDTRnoRosa-preDTXJan31shuffle1_1030000_filtered.h5",
        # "./aging/12mo/1yr-dtr_norosa-2/1yr-dtr_norosa-2-predtx/1yrDTRnoRosa-M2-preDTX-23102023_000017DLC_resnet50_1yrDTRnoRosa-preDTXJan31shuffle1_1030000_filtered.h5",
    ]
    age_dtrpre_sin = [
        "./aging/12mo/1yr-dtr_norosa-1/1yr-dtr_norosa-1-predtx/1yrDTRnoRosa-M1-19102023_000018DLC_resnet50_1yrDTRnoRosa-preDTXJan31shuffle1_1030000_filtered.h5",
        # "./aging/12mo/1yr-dtr_norosa-1/1yr-dtr_norosa-1-predtx/1yrDTRnoRosa-M1-19102023_000017DLC_resnet50_1yrDTRnoRosa-preDTXJan31shuffle1_1030000_filtered.h5",
        "./aging/12mo/1yr-dtr_norosa-2/1yr-dtr_norosa-2-predtx/1yrDTRnoRosa-M2-preDTX-23102023_000019DLC_resnet50_1yrDTRnoRosa-preDTXJan31shuffle1_1030000_filtered.h5",
        # "./aging/12mo/1yr-dtr_norosa-2/1yr-dtr_norosa-2-predtx/1yrDTRnoRosa-M2-preDTX-23102023_000020DLC_resnet50_1yrDTRnoRosa-preDTXJan31shuffle1_1030000_filtered.h5",
    ]

    # Adding 12m Aged Post-DTX Mice
    age_dtrpost_non = [
        "./aging/12mo/1yr-dtr_norosa-1/1yr-dtr_norosa-1-postdtx/1yrDTRnoRosa-M1-postDTX-31102023_000000DLC_resnet50_1yrDTRnoRosa-postDTXFeb13shuffle1_700000_filtered.h5",
        "./aging/12mo/1yr-dtr_norosa-2/1yr-dtr_norosa-2-postdtx/1yrDTRnoRosa-M2-postDTX-01112023_000002DLC_resnet50_1yrDTRnoRosa-postDTXFeb13shuffle1_700000_filtered.h5",
    ]
    age_dtrpost_per = [
        "./aging/12mo/1yr-dtr_norosa-1/1yr-dtr_norosa-1-postdtx/1yrDTRnoRosa-M1-postDTX-31102023_000007DLC_resnet50_1yrDTRnoRosa-postDTXFeb13shuffle1_700000_filtered.h5",
        "./aging/12mo/1yr-dtr_norosa-2/1yr-dtr_norosa-2-postdtx/1yrDTRnoRosa-M2-postDTX-01112023_000011DLC_resnet50_1yrDTRnoRosa-postDTXFeb13shuffle1_700000_filtered.h5",
    ]
    age_dtrpost_sin = [
        "./aging/12mo/1yr-dtr_norosa-1/1yr-dtr_norosa-1-postdtx/1yrDTRnoRosa-M1-postDTX-31102023_000014DLC_resnet50_1yrDTRnoRosa-postDTXFeb13shuffle1_700000_filtered.h5",
        "./aging/12mo/1yr-dtr_norosa-2/1yr-dtr_norosa-2-postdtx/1yrDTRnoRosa-M2-postDTX-01112023_000018DLC_resnet50_1yrDTRnoRosa-postDTXFeb13shuffle1_700000_filtered.h5",
    ]

    # 18m Old Mice
    old_dtrpre_non = [
        "./aging/18mo/18mo-RAP-predtx/1-6yrRAP-M1-preDTX_000000DLC_resnet50_dtr_update_predtxApr8shuffle1_1110000_filtered.h5",
        # "./aging/18mo/18mo-RAP-predtx/1-6yrRAP-M1-preDTX_000001DLC_resnet50_dtr_update_predtxApr8shuffle1_1110000_filtered.h5",
        # "./aging/18mo/18mo-RAP-predtx/1-6yrRAP-M1-preDTX_000002DLC_resnet50_dtr_update_predtxApr8shuffle1_1110000_filtered.h5",
    ]
    old_dtrpre_per = [
        # "./aging/18mo/18mo-RAP-predtx/1-6yrRAP-M1-preDTX_000013DLC_resnet50_dtr_update_predtxApr8shuffle1_1110000_filtered.h5",
        "./aging/18mo/18mo-RAP-predtx/1-6yrRAP-M1-preDTX_000014DLC_resnet50_dtr_update_predtxApr8shuffle1_1110000_filtered.h5",
    ]
    old_dtrpre_sin = [
        # "./aging/18mo/18mo-RAP-predtx/1-6yrRAP-M1-preDTX_000019DLC_resnet50_dtr_update_predtxApr8shuffle1_1110000_filtered.h5",
        "./aging/18mo/18mo-RAP-predtx/1-6yrRAP-M1-preDTX_000020DLC_resnet50_dtr_update_predtxApr8shuffle1_1110000_filtered.h5",
    ]

    mos_df = df(columns=["Condition", "Limb", "Perturbation State", "MoS (cm)"])

    mos_df = condition_add_adj(
        mos_df,
        wt_non_lmos,
        wt_non,
        wt_x_channels,
        wt_y_channels,
        "WT",
        "Left",
        "Non-Perturbation",
    )
    mos_df = condition_add_adj(
        mos_df,
        wt_non_rmos,
        wt_non,
        wt_x_channels,
        wt_y_channels,
        "WT",
        "Right",
        "Non-Perturbation",
    )
    mos_df = condition_add_adj(
        mos_df,
        wt_per_lmos,
        wt_per,
        wt_x_channels,
        wt_y_channels,
        "WT",
        "Left",
        "Perturbation",
    )
    mos_df = condition_add_adj(
        mos_df,
        wt_per_rmos,
        wt_per,
        wt_x_channels,
        wt_y_channels,
        "WT",
        "Right",
        "Perturbation",
    )
    mos_df = condition_add_adj(
        mos_df,
        wt_sin_lmos,
        wt_sin,
        wt_x_channels,
        wt_y_channels,
        "WT",
        "Left",
        "Sinusoidal",
    )
    mos_df = condition_add_adj(
        mos_df,
        wt_sin_rmos,
        wt_sin,
        wt_x_channels,
        wt_y_channels,
        "WT",
        "Right",
        "Sinusoidal",
    )

    mos_df = condition_add_adj(
        mos_df,
        egr3_non_lmos,
        egr3_non,
        wt_x_channels,
        wt_y_channels,
        "Egr3",
        "Left",
        "Non-Perturbation",
    )
    mos_df = condition_add_adj(
        mos_df,
        egr3_non_rmos,
        egr3_non,
        wt_x_channels,
        wt_y_channels,
        "Egr3",
        "Right",
        "Non-Perturbation",
    )
    mos_df = condition_add_adj(
        mos_df,
        egr3_per_lmos,
        egr3_per,
        wt_x_channels,
        wt_y_channels,
        "Egr3",
        "Left",
        "Perturbation",
    )
    mos_df = condition_add_adj(
        mos_df,
        egr3_per_rmos,
        egr3_per,
        wt_x_channels,
        wt_y_channels,
        "Egr3",
        "Right",
        "Perturbation",
    )
    mos_df = condition_add_adj(
        mos_df,
        egr3_sin_lmos,
        egr3_sin,
        wt_x_channels,
        wt_y_channels,
        "Egr3",
        "Left",
        "Sinusoidal",
    )
    mos_df = condition_add_adj(
        mos_df,
        egr3_sin_rmos,
        egr3_sin,
        wt_x_channels,
        wt_y_channels,
        "Egr3",
        "Right",
        "Sinusoidal",
    )

    # Adding young DTR mice
    mos_df = condition_add_adj(
        mos_df,
        dtrpre_non_lmos,
        dtrpre_non,
        dtr_x_channels,
        dtr_y_channels,
        "4m Pre-DTX",
        "Left",
        "Non-Perturbation",
    )
    mos_df = condition_add_adj(
        mos_df,
        dtrpre_non_rmos,
        dtrpre_non,
        dtr_x_channels,
        dtr_y_channels,
        "4m Pre-DTX",
        "Right",
        "Non-Perturbation",
    )
    mos_df = condition_add_adj(
        mos_df,
        dtrpre_per_lmos,
        dtrpre_per,
        dtr_x_channels,
        dtr_y_channels,
        "4m Pre-DTX",
        "Left",
        "Perturbation",
    )
    mos_df = condition_add_adj(
        mos_df,
        dtrpre_per_rmos,
        dtrpre_per,
        dtr_x_channels,
        dtr_y_channels,
        "4m Pre-DTX",
        "Right",
        "Perturbation",
    )
    mos_df = condition_add_adj(
        mos_df,
        dtrpre_sin_lmos,
        dtrpre_sin,
        dtr_x_channels,
        dtr_y_channels,
        "4m Pre-DTX",
        "Left",
        "Sinusoidal",
    )
    mos_df = condition_add_adj(
        mos_df,
        dtrpre_sin_rmos,
        dtrpre_sin,
        dtr_x_channels,
        dtr_y_channels,
        "4m Pre-DTX",
        "Right",
        "Sinusoidal",
    )

    mos_df = condition_add_adj(
        mos_df,
        dtrpost_non_lmos,
        dtrpost_non,
        dtr_x_channels,
        dtr_y_channels,
        "4m Post-DTX",
        "Left",
        "Non-Perturbation",
    )
    mos_df = condition_add_adj(
        mos_df,
        dtrpost_non_rmos,
        dtrpost_non,
        dtr_x_channels,
        dtr_y_channels,
        "4m Post-DTX",
        "Right",
        "Non-Perturbation",
    )
    mos_df = condition_add_adj(
        mos_df,
        dtrpost_per_lmos,
        dtrpost_per,
        dtr_x_channels,
        dtr_y_channels,
        "4m Post-DTX",
        "Left",
        "Perturbation",
    )
    mos_df = condition_add_adj(
        mos_df,
        dtrpost_per_rmos,
        dtrpost_per,
        dtr_x_channels,
        dtr_y_channels,
        "4m Post-DTX",
        "Right",
        "Perturbation",
    )
    mos_df = condition_add_adj(
        mos_df,
        dtrpost_sin_lmos,
        dtrpost_sin,
        dtr_x_channels,
        dtr_y_channels,
        "4m Post-DTX",
        "Left",
        "Sinusoidal",
    )
    mos_df = condition_add_adj(
        mos_df,
        dtrpost_sin_rmos,
        dtrpost_sin,
        dtr_x_channels,
        dtr_y_channels,
        "4m Post-DTX",
        "Right",
        "Sinusoidal",
    )

    # Adding 12mo aged DTR mice
    mos_df = condition_add_adj_dlc(
        mos_df,
        age_predtx_non_lmos,
        age_dtrpre_non,
        "12m Pre-DTX",
        "Left",
        "Non-Perturbation",
    )
    mos_df = condition_add_adj_dlc(
        mos_df,
        age_predtx_non_rmos,
        age_dtrpre_non,
        "12m Pre-DTX",
        "Right",
        "Non-Perturbation",
    )
    mos_df = condition_add_adj_dlc(
        mos_df,
        age_predtx_per_lmos,
        age_dtrpre_per,
        "12m Pre-DTX",
        "Left",
        "Perturbation",
    )
    mos_df = condition_add_adj_dlc(
        mos_df,
        age_predtx_per_rmos,
        age_dtrpre_per,
        "12m Pre-DTX",
        "Right",
        "Perturbation",
    )
    mos_df = condition_add_adj_dlc(
        mos_df, age_predtx_sin_lmos, age_dtrpre_sin, "12m Pre-DTX", "Left", "Sinusoidal"
    )
    mos_df = condition_add_adj_dlc(
        mos_df,
        age_predtx_sin_rmos,
        age_dtrpre_sin,
        "12m Pre-DTX",
        "Right",
        "Sinusoidal",
    )

    # Adding 12m Post-DTX group
    mos_df = condition_add_adj_dlc(
        mos_df,
        age_postdtx_non_lmos,
        age_dtrpost_non,
        "12m Post-DTX",
        "Left",
        "Non-Perturbation",
    )
    mos_df = condition_add_adj_dlc(
        mos_df,
        age_postdtx_non_rmos,
        age_dtrpost_non,
        "12m Post-DTX",
        "Right",
        "Non-Perturbation",
    )
    mos_df = condition_add_adj_dlc(
        mos_df,
        age_postdtx_per_lmos,
        age_dtrpost_per,
        "12m Post-DTX",
        "Left",
        "Perturbation",
    )
    mos_df = condition_add_adj_dlc(
        mos_df,
        age_postdtx_per_rmos,
        age_dtrpost_per,
        "12m Post-DTX",
        "Right",
        "Perturbation",
    )
    mos_df = condition_add_adj_dlc(
        mos_df,
        age_postdtx_sin_lmos,
        age_dtrpost_sin,
        "12m Post-DTX",
        "Left",
        "Sinusoidal",
    )
    mos_df = condition_add_adj_dlc(
        mos_df,
        age_postdtx_sin_rmos,
        age_dtrpost_sin,
        "12m Post-DTX",
        "Right",
        "Sinusoidal",
    )

    # Adding 18mo old group
    mos_df = condition_add_adj_dlc(
        mos_df,
        old_predtx_non_lmos,
        old_dtrpre_non,
        "18m Pre-DTX",
        "Left",
        "Non-Perturbation",
    )
    mos_df = condition_add_adj_dlc(
        mos_df,
        old_predtx_non_rmos,
        old_dtrpre_non,
        "18m Pre-DTX",
        "Right",
        "Non-Perturbation",
    )
    mos_df = condition_add_adj_dlc(
        mos_df,
        old_predtx_per_lmos,
        old_dtrpre_per,
        "18m Pre-DTX",
        "Left",
        "Perturbation",
    )
    mos_df = condition_add_adj_dlc(
        mos_df,
        old_predtx_per_rmos,
        old_dtrpre_per,
        "18m Pre-DTX",
        "Right",
        "Perturbation",
    )
    mos_df = condition_add_adj_dlc(
        mos_df, old_predtx_sin_lmos, old_dtrpre_sin, "18m Pre-DTX", "Left", "Sinusoidal"
    )
    mos_df = condition_add_adj_dlc(
        mos_df,
        old_predtx_sin_rmos,
        old_dtrpre_sin,
        "18m Pre-DTX",
        "Right",
        "Sinusoidal",
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
        style="white",
        font="serif",
        font_scale=1.6,
        palette="colorblind",
        rc=custom_params,
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
        "inner": "point",
    }

    # axs[0].set_title("MoS between conditions")
    cond_combo_comp = sns.violinplot(**cond_combo_plot_params, ci=95, capsize=0.05)
    plt.legend(loc="upper left", fontsize=16)
    annotator = Annotator(cond_combo_comp, condition_pairs, **cond_combo_plot_params)
    annotator.new_plot(
        cond_combo_comp, condition_pairs, plot="violinplot", **cond_combo_plot_params
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
        print("Results Saved")
    else:
        print("Results not saved")

    plt.show()


if __name__ == "__main__":
    main()
