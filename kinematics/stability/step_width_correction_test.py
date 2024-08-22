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


def sw_mos_correction(mos_values, fl_sw, hl_sw):
    avg_sw = (np.mean(fl_sw) + np.mean(hl_sw)) / 2
    print(avg_sw)


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


def main():

    # Some Parameters about what to do
    save_stepw = False
    save_hiph = False

    step_width_df = df(
        columns=["Condition", "Limb", "Perturbation State", "Step Width (cm)"]
    )
    hiph_df = df(columns=["Condition", "Perturbation State", "Hip Height (cm)"])

    conditions = [
        "Non-Perturbation",
        "Perturbation",
        "Sinusoidal",
    ]

    # NOTE: WT channel names also work for Egr3
    # wt_y_channels = ["35 FRy (cm)", "33 FLy (cm)", "30 HRy (cm)", "28 HLy (cm)"]
    # wt_x_channels = ["34 FRx (cm)", "32 FLx (cm)", "29 HRx (cm)", "27 HLx (cm)"]
    #
    # wt_fl_x_channels = ["34 FRx (cm)", "32 FLx (cm)"]
    # wt_fl_y_channels = ["35 FRy (cm)", "33 FLy (cm)"]
    # wt_hl_x_channels = ["29 HRx (cm)", "27 HLx (cm)"]
    # wt_hl_y_channels = ["30 HRy (cm)", "28 HLy (cm)"]
    #
    # wt_hh_channels = ["24 toey (cm)", "16 Hipy (cm)"]

    dtr_fl_x_channels = ["39 FRx (cm)", "37 FLx (cm)"]
    dtr_fl_y_channels = ["40 FRy (cm)", "38 FLy (cm)"]
    dtr_hl_x_channels = ["35 HRx (cm)", "33 HLx (cm)"]
    dtr_hl_y_channels = ["36 HRy (cm)", "34 HLy (cm)"]

    dtr_hh_channels = ["25 toey (cm)", "17 Hipy (cm)"]

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
    ]
    egr3_per = [
        "./egr3_data/egr3-6-per-all.txt",
        "./egr3_data/egr3-7-per-all.txt",
        "./egr3_data/egr3-8-per-all.txt",
        "./egr3_data/egr3-9-per-all-2.txt",
    ]
    egr3_sin = [
        "./egr3_data/egr3-6-sin-all.txt",
        "./egr3_data/egr3-7-sin-all.txt",
        "./egr3_data/egr3-8-sin-all.txt",
        "./egr3_data/egr3-9-sin-all-1.txt",
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
        "./dtr_data/predtx/dtr-pre-3-sin-all.txt",
        "./dtr_data/predtx/dtr-pre-5-sin-all.txt",
        "./dtr_data/predtx/dtr-pre-6-sin-all.txt",
        "./dtr_data/predtx/dtr-pre-7-sin-all.txt",
    ]

    dtrpost_non = [
        "./dtr_data/postdtx/dtr-post-2-non-all.txt",
        "./dtr_data/postdtx/dtr-post-3-non-all.txt",
        "./dtr_data/postdtx/dtr-post-5-non-all.txt",
        "./dtr_data/postdtx/dtr-post-6-non-all.txt",
        "./dtr_data/postdtx/dtr-post-8-non-all.txt",
    ]
    dtrpost_per = [
        "./dtr_data/postdtx/dtr-post-2-per-all.txt",
        "./dtr_data/postdtx/dtr-post-3-per-all.txt",
        "./dtr_data/postdtx/dtr-post-5-per-all.txt",
        "./dtr_data/postdtx/dtr-post-6-per-all.txt",
    ]
    dtrpost_sin = [
        "./dtr_data/postdtx/dtr-post-2-sin-all.txt",
        "./dtr_data/postdtx/dtr-post-3-sin-all.txt",
        "./dtr_data/postdtx/dtr-post-5-sin-all.txt",
        "./dtr_data/postdtx/dtr-post-6-sin-all.txt",
    ]

    # Adding 12m Aged Mice
    age_dtrpre_non = [
        "./aging/12mo/1yr-dtr_norosa-1/1yr-dtr_norosa-1-predtx/1yrDTRnoRosa-M1-19102023_000000DLC_resnet50_1yrDTRnoRosa-preDTXJan31shuffle1_1030000_filtered.h5",
        "./aging/12mo/1yr-dtr_norosa-2/1yr-dtr_norosa-2-predtx/1yrDTRnoRosa-M2-preDTX-23102023_000000DLC_resnet50_1yrDTRnoRosa-preDTXJan31shuffle1_1030000_filtered.h5",
        "./aging/12mo/1yr-dtr_norosa-2/1yr-dtr_norosa-2-predtx/1yrDTRnoRosa-M2-preDTX-23102023_000001DLC_resnet50_1yrDTRnoRosa-preDTXJan31shuffle1_1030000_filtered.h5",
    ]
    age_dtrpre_per = [
        "./aging/12mo/1yr-dtr_norosa-1/1yr-dtr_norosa-1-predtx/1yrDTRnoRosa-M1-19102023_000009DLC_resnet50_1yrDTRnoRosa-preDTXJan31shuffle1_1030000_filtered.h5",
        "./aging/12mo/1yr-dtr_norosa-1/1yr-dtr_norosa-1-predtx/1yrDTRnoRosa-M1-19102023_000010DLC_resnet50_1yrDTRnoRosa-preDTXJan31shuffle1_1030000_filtered.h5",
        "./aging/12mo/1yr-dtr_norosa-2/1yr-dtr_norosa-2-predtx/1yrDTRnoRosa-M2-preDTX-23102023_000016DLC_resnet50_1yrDTRnoRosa-preDTXJan31shuffle1_1030000_filtered.h5",
        "./aging/12mo/1yr-dtr_norosa-2/1yr-dtr_norosa-2-predtx/1yrDTRnoRosa-M2-preDTX-23102023_000017DLC_resnet50_1yrDTRnoRosa-preDTXJan31shuffle1_1030000_filtered.h5",
    ]
    age_dtrpre_sin = [
        "./aging/12mo/1yr-dtr_norosa-1/1yr-dtr_norosa-1-predtx/1yrDTRnoRosa-M1-19102023_000018DLC_resnet50_1yrDTRnoRosa-preDTXJan31shuffle1_1030000_filtered.h5",
        "./aging/12mo/1yr-dtr_norosa-1/1yr-dtr_norosa-1-predtx/1yrDTRnoRosa-M1-19102023_000017DLC_resnet50_1yrDTRnoRosa-preDTXJan31shuffle1_1030000_filtered.h5",
        "./aging/12mo/1yr-dtr_norosa-2/1yr-dtr_norosa-2-predtx/1yrDTRnoRosa-M2-preDTX-23102023_000019DLC_resnet50_1yrDTRnoRosa-preDTXJan31shuffle1_1030000_filtered.h5",
        "./aging/12mo/1yr-dtr_norosa-2/1yr-dtr_norosa-2-predtx/1yrDTRnoRosa-M2-preDTX-23102023_000020DLC_resnet50_1yrDTRnoRosa-preDTXJan31shuffle1_1030000_filtered.h5",
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
        "./aging/18mo/18mo-RAP-predtx/1-6yrRAP-M1-preDTX_000001DLC_resnet50_dtr_update_predtxApr8shuffle1_1110000_filtered.h5",
        "./aging/18mo/18mo-RAP-predtx/1-6yrRAP-M1-preDTX_000002DLC_resnet50_dtr_update_predtxApr8shuffle1_1110000_filtered.h5",
    ]
    old_dtrpre_per = [
        "./aging/18mo/18mo-RAP-predtx/1-6yrRAP-M1-preDTX_000013DLC_resnet50_dtr_update_predtxApr8shuffle1_1110000_filtered.h5",
        "./aging/18mo/18mo-RAP-predtx/1-6yrRAP-M1-preDTX_000014DLC_resnet50_dtr_update_predtxApr8shuffle1_1110000_filtered.h5",
    ]
    old_dtrpre_sin = [
        "./aging/18mo/18mo-RAP-predtx/1-6yrRAP-M1-preDTX_000019DLC_resnet50_dtr_update_predtxApr8shuffle1_1110000_filtered.h5",
        "./aging/18mo/18mo-RAP-predtx/1-6yrRAP-M1-preDTX_000020DLC_resnet50_dtr_update_predtxApr8shuffle1_1110000_filtered.h5",
    ]

    # Adding Step Widths
    # step_width_df = sw_condition_add(
    #     step_width_df,
    #     wt_non,
    #     wt_fl_x_channels,
    #     wt_fl_y_channels,
    #     "WT",
    #     "Forelimb",
    #     "Non-Perturbation",
    # )
    # step_width_df = sw_condition_add(
    #     step_width_df,
    #     wt_non,
    #     wt_hl_x_channels,
    #     wt_hl_y_channels,
    #     "WT",
    #     "Hindlimb",
    #     "Non-Perturbation",
    # )
    # step_width_df = sw_condition_add(
    #     step_width_df,
    #     wt_per,
    #     wt_fl_x_channels,
    #     wt_fl_y_channels,
    #     "WT",
    #     "Forelimb",
    #     "Perturbation",
    # )
    # step_width_df = sw_condition_add(
    #     step_width_df,
    #     wt_per,
    #     wt_hl_x_channels,
    #     wt_hl_y_channels,
    #     "WT",
    #     "Hindlimb",
    #     "Perturbation",
    # )
    # step_width_df = sw_condition_add(
    #     step_width_df,
    #     wt_sin,
    #     wt_fl_x_channels,
    #     wt_fl_y_channels,
    #     "WT",
    #     "Forelimb",
    #     "Sinusoidal",
    # )
    # step_width_df = sw_condition_add(
    #     step_width_df,
    #     wt_sin,
    #     wt_hl_x_channels,
    #     wt_hl_y_channels,
    #     "WT",
    #     "Hindlimb",
    #     "Sinusoidal",
    # )
    #
    # step_width_df = sw_condition_add(
    #     step_width_df,
    #     egr3_non,
    #     wt_fl_x_channels,
    #     wt_fl_y_channels,
    #     "Egr3",
    #     "Forelimb",
    #     "Non-Perturbation",
    # )
    # step_width_df = sw_condition_add(
    #     step_width_df,
    #     egr3_non,
    #     wt_hl_x_channels,
    #     wt_hl_y_channels,
    #     "Egr3",
    #     "Hindlimb",
    #     "Non-Perturbation",
    # )
    # step_width_df = sw_condition_add(
    #     step_width_df,
    #     egr3_per,
    #     wt_fl_x_channels,
    #     wt_fl_y_channels,
    #     "Egr3",
    #     "Forelimb",
    #     "Perturbation",
    # )
    # step_width_df = sw_condition_add(
    #     step_width_df,
    #     egr3_per,
    #     wt_hl_x_channels,
    #     wt_hl_y_channels,
    #     "Egr3",
    #     "Hindlimb",
    #     "Perturbation",
    # )
    # step_width_df = sw_condition_add(
    #     step_width_df,
    #     egr3_sin,
    #     wt_fl_x_channels,
    #     wt_fl_y_channels,
    #     "Egr3",
    #     "Forelimb",
    #     "Sinusoidal",
    # )
    # step_width_df = sw_condition_add(
    #     step_width_df,
    #     egr3_sin,
    #     wt_hl_x_channels,
    #     wt_hl_y_channels,
    #     "Egr3",
    #     "Hindlimb",
    #     "Sinusoidal",
    # )

    step_width_df = sw_condition_add(
        step_width_df,
        dtrpre_non,
        dtr_fl_x_channels,
        dtr_fl_y_channels,
        "Pre-DTX",
        "Forelimb",
        "Non-Perturbation",
    )
    step_width_df = sw_condition_add(
        step_width_df,
        dtrpre_non,
        dtr_hl_x_channels,
        dtr_hl_y_channels,
        "Pre-DTX",
        "Hindlimb",
        "Non-Perturbation",
    )
    step_width_df = sw_condition_add(
        step_width_df,
        dtrpre_per,
        dtr_fl_x_channels,
        dtr_fl_y_channels,
        "Pre-DTX",
        "Forelimb",
        "Perturbation",
    )
    step_width_df = sw_condition_add(
        step_width_df,
        dtrpre_per,
        dtr_hl_x_channels,
        dtr_hl_y_channels,
        "Pre-DTX",
        "Hindlimb",
        "Perturbation",
    )
    step_width_df = sw_condition_add(
        step_width_df,
        dtrpre_sin,
        dtr_fl_x_channels,
        dtr_fl_y_channels,
        "Pre-DTX",
        "Forelimb",
        "Sinusoidal",
    )
    step_width_df = sw_condition_add(
        step_width_df,
        dtrpre_sin,
        dtr_hl_x_channels,
        dtr_hl_y_channels,
        "Pre-DTX",
        "Hindlimb",
        "Sinusoidal",
    )

    # step_width_df = sw_condition_add(
    #     step_width_df,
    #     dtrpost_non,
    #     dtr_fl_x_channels,
    #     dtr_fl_y_channels,
    #     "Post-DTX",
    #     "Forelimb",
    #     "Non-Perturbation",
    # )
    # step_width_df = sw_condition_add(
    #     step_width_df,
    #     dtrpost_non,
    #     dtr_hl_x_channels,
    #     dtr_hl_y_channels,
    #     "Post-DTX",
    #     "Hindlimb",
    #     "Non-Perturbation",
    # )
    # step_width_df = sw_condition_add(
    #     step_width_df,
    #     dtrpost_per,
    #     dtr_fl_x_channels,
    #     dtr_fl_y_channels,
    #     "Post-DTX",
    #     "Forelimb",
    #     "Perturbation",
    # )
    # step_width_df = sw_condition_add(
    #     step_width_df,
    #     dtrpost_per,
    #     dtr_hl_x_channels,
    #     dtr_hl_y_channels,
    #     "Post-DTX",
    #     "Hindlimb",
    #     "Perturbation",
    # )
    # step_width_df = sw_condition_add(
    #     step_width_df,
    #     dtrpost_sin,
    #     dtr_fl_x_channels,
    #     dtr_fl_y_channels,
    #     "Post-DTX",
    #     "Forelimb",
    #     "Sinusoidal",
    # )
    # step_width_df = sw_condition_add(
    #     step_width_df,
    #     dtrpost_sin,
    #     dtr_hl_x_channels,
    #     dtr_hl_y_channels,
    #     "Post-DTX",
    #     "Hindlimb",
    #     "Sinusoidal",
    # )

    # 12m aged mice

    # 12m Pre-DTX
    step_width_df = dlc_sw_cond_add(
        step_width_df, age_dtrpre_non, "12m Pre-DTX", "Forelimb", "Non-Perturbation"
    )
    step_width_df = dlc_sw_cond_add(
        step_width_df, age_dtrpre_non, "12m Pre-DTX", "Hindlimb", "Non-Perturbation"
    )
    step_width_df = dlc_sw_cond_add(
        step_width_df, age_dtrpre_per, "12m Pre-DTX", "Forelimb", "Perturbation"
    )
    step_width_df = dlc_sw_cond_add(
        step_width_df, age_dtrpre_per, "12m Pre-DTX", "Hindlimb", "Perturbation"
    )
    step_width_df = dlc_sw_cond_add(
        step_width_df, age_dtrpre_sin, "12m Pre-DTX", "Forelimb", "Sinusoidal"
    )
    step_width_df = dlc_sw_cond_add(
        step_width_df, age_dtrpre_sin, "12m Pre-DTX", "Hindlimb", "Sinusoidal"
    )

    # 12m Post-DTX
    step_width_df = dlc_sw_cond_add(
        step_width_df, age_dtrpost_non, "12m Post-DTX", "Forelimb", "Non-Perturbation"
    )
    step_width_df = dlc_sw_cond_add(
        step_width_df, age_dtrpost_non, "12m Post-DTX", "Hindlimb", "Non-Perturbation"
    )
    step_width_df = dlc_sw_cond_add(
        step_width_df, age_dtrpost_per, "12m Post-DTX", "Forelimb", "Perturbation"
    )
    step_width_df = dlc_sw_cond_add(
        step_width_df, age_dtrpost_per, "12m Post-DTX", "Hindlimb", "Perturbation"
    )
    step_width_df = dlc_sw_cond_add(
        step_width_df, age_dtrpost_sin, "12m Post-DTX", "Forelimb", "Sinusoidal"
    )
    step_width_df = dlc_sw_cond_add(
        step_width_df, age_dtrpost_sin, "12m Post-DTX", "Hindlimb", "Sinusoidal"
    )

    # Older 18m
    step_width_df = dlc_sw_cond_add(
        step_width_df, old_dtrpre_non, "18m Pre-DTX", "Forelimb", "Non-Perturbation"
    )
    step_width_df = dlc_sw_cond_add(
        step_width_df, old_dtrpre_non, "18m Pre-DTX", "Hindlimb", "Non-Perturbation"
    )
    step_width_df = dlc_sw_cond_add(
        step_width_df, old_dtrpre_per, "18m Pre-DTX", "Forelimb", "Perturbation"
    )
    step_width_df = dlc_sw_cond_add(
        step_width_df, old_dtrpre_per, "18m Pre-DTX", "Hindlimb", "Perturbation"
    )
    step_width_df = dlc_sw_cond_add(
        step_width_df, old_dtrpre_sin, "18m Pre-DTX", "Forelimb", "Sinusoidal"
    )
    step_width_df = dlc_sw_cond_add(
        step_width_df, old_dtrpre_sin, "18m Pre-DTX", "Hindlimb", "Sinusoidal"
    )

    # Now doing Hip Height

    # WT
    # hiph_df = hiph_condition_add(
    #     hiph_df, wt_non, wt_hh_channels, "WT", "Non-Perturbation"
    # )
    # hiph_df = hiph_condition_add(hiph_df, wt_per, wt_hh_channels, "WT", "Perturbation")
    # hiph_df = hiph_condition_add(hiph_df, wt_sin, wt_hh_channels, "WT", "Sinusoidal")
    #
    # # Egr3
    # hiph_df = hiph_condition_add(
    #     hiph_df, egr3_non, wt_hh_channels, "Egr3", "Non-Perturbation"
    # )
    # hiph_df = hiph_condition_add(
    #     hiph_df, egr3_per, wt_hh_channels, "Egr3", "Perturbation"
    # )
    # hiph_df = hiph_condition_add(
    #     hiph_df, egr3_sin, wt_hh_channels, "Egr3", "Sinusoidal"
    # )

    # Pre-DTX
    hiph_df = hiph_condition_add(
        hiph_df, dtrpre_non, dtr_hh_channels, "Pre-DTX", "Non-Perturbation"
    )
    hiph_df = hiph_condition_add(
        hiph_df, dtrpre_per, dtr_hh_channels, "Pre-DTX", "Perturbation"
    )
    hiph_df = hiph_condition_add(
        hiph_df, dtrpre_sin, dtr_hh_channels, "Pre-DTX", "Sinusoidal"
    )

    # Post-DTX
    # hiph_df = hiph_condition_add(
    #     hiph_df, dtrpost_non, dtr_hh_channels, "Post-DTX", "Non-Perturbation"
    # )
    # hiph_df = hiph_condition_add(
    #     hiph_df, dtrpost_per, dtr_hh_channels, "Post-DTX", "Perturbation"
    # )
    # hiph_df = hiph_condition_add(
    #     hiph_df, dtrpost_sin, dtr_hh_channels, "Post-DTX", "Sinusoidal"
    # )

    # 12m Mice

    # 12m Mice Pre-DTX
    hiph_df = dlc_hiph_cond_add(
        hiph_df, age_dtrpre_non, "12m Pre-DTX", "Non-Perturbation"
    )
    hiph_df = dlc_hiph_cond_add(hiph_df, age_dtrpre_per, "12m Pre-DTX", "Perturbation")
    hiph_df = dlc_hiph_cond_add(hiph_df, age_dtrpre_sin, "12m Pre-DTX", "Sinusoidal")

    # 12m Mice Post-DTX
    hiph_df = dlc_hiph_cond_add(
        hiph_df, age_dtrpost_non, "12m Post-DTX", "Non-Perturbation"
    )
    hiph_df = dlc_hiph_cond_add(
        hiph_df, age_dtrpost_per, "12m Post-DTX", "Perturbation"
    )
    hiph_df = dlc_hiph_cond_add(hiph_df, age_dtrpost_sin, "12m Post-DTX", "Sinusoidal")

    # 18m Mice
    hiph_df = dlc_hiph_cond_add(
        hiph_df, old_dtrpre_non, "18m Pre-DTX", "Non-Perturbation"
    )
    hiph_df = dlc_hiph_cond_add(hiph_df, old_dtrpre_per, "18m Pre-DTX", "Perturbation")
    hiph_df = dlc_hiph_cond_add(hiph_df, old_dtrpre_sin, "18m Pre-DTX", "Sinusoidal")

    sw_combo = step_width_df.drop(columns=["Limb"])
    con_sw_combo = step_width_df.drop(columns=["Limb"])

    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set(
        style="white",
        font="serif",
        font_scale=1.7,
        palette="colorblind",
        rc=custom_params,
    )

    # fig, axs = plt.subplots(2)

    combo_pairs = [
        [("Non-Perturbation"), ("Perturbation")],
    ]
    combo_legend = ["Non-Perturbation", "Perturbation", "Sinusoidal"]

    perturbation_state_order = ["Non-Perturbation", "Perturbation", "Sinusoidal"]

    # plt.title("Step Width between WT, Egr3 KO, and DTX Mice Pre and Post Injection")
    condition_pairs = [
        # Comparison within wildtype condition
        # [("WT", "Non-Perturbation"), ("WT", "Perturbation")],
        # [("WT", "Sinusoidal"), ("WT", "Perturbation")],
        # [("WT", "Non-Perturbation"), ("WT", "Sinusoidal")],
        # # Comparison between Wildtype and Pre-DTX
        # [("WT", "Non-Perturbation"), ("Pre-DTX", "Non-Perturbation")],
        # [("WT", "Sinusoidal"), ("Pre-DTX", "Sinusoidal")],
        # [("WT", "Perturbation"), ("Pre-DTX", "Perturbation")],
        # # Comparison within Egr3 condition
        # [("Egr3", "Non-Perturbation"), ("Egr3", "Perturbation")],
        # [("Egr3", "Sinusoidal"), ("Egr3", "Perturbation")],
        # [("Egr3", "Non-Perturbation"), ("Egr3", "Sinusoidal")],
        # Comparison within Pre-DTX condition
        [("Pre-DTX", "Non-Perturbation"), ("Pre-DTX", "Perturbation")],
        [("Pre-DTX", "Sinusoidal"), ("Pre-DTX", "Perturbation")],
        [("Pre-DTX", "Non-Perturbation"), ("Pre-DTX", "Sinusoidal")],
        # Comparison within Post-DTX condition
        # [("Post-DTX", "Non-Perturbation"), ("Post-DTX", "Perturbation")],
        # [("Post-DTX", "Sinusoidal"), ("Post-DTX", "Perturbation")],
        # [("Post-DTX", "Non-Perturbation"), ("Post-DTX", "Sinusoidal")],
        # Comparison Between DTX conditions
        # [("Pre-DTX", "Non-Perturbation"), ("Post-DTX", "Non-Perturbation")],
        # [("Pre-DTX", "Sinusoidal"), ("Post-DTX", "Sinusoidal")],
        # [("Pre-DTX", "Perturbation"), ("Post-DTX", "Perturbation")],
        # Comparison within 12m conditions
        [("12m Pre-DTX", "Non-Perturbation"), ("12m Pre-DTX", "Perturbation")],
        [("12m Pre-DTX", "Sinusoidal"), ("12m Pre-DTX", "Perturbation")],
        [("12m Pre-DTX", "Non-Perturbation"), ("12m Pre-DTX", "Sinusoidal")],
        # Comparison Between Ages Pre-DTX conditions
        [("Pre-DTX", "Non-Perturbation"), ("12m Pre-DTX", "Non-Perturbation")],
        [("Pre-DTX", "Sinusoidal"), ("12m Pre-DTX", "Sinusoidal")],
        [("Pre-DTX", "Perturbation"), ("12m Pre-DTX", "Perturbation")],
        # Comparison Between Ages Pre-DTX conditions
        [("Pre-DTX", "Non-Perturbation"), ("18m Pre-DTX", "Non-Perturbation")],
        [("Pre-DTX", "Sinusoidal"), ("18m Pre-DTX", "Sinusoidal")],
        [("Pre-DTX", "Perturbation"), ("18m Pre-DTX", "Perturbation")],
    ]

    perturbation_state_order = ["Non-Perturbation", "Perturbation", "Sinusoidal"]

    sw_plot_params = {
        "data": con_sw_combo,
        "x": "Condition",
        "y": "Step Width (cm)",
        "hue": "Perturbation State",
        "hue_order": perturbation_state_order,
        # "inner": "point",
    }

    hiph_plot_params = {
        "data": hiph_df,
        "x": "Condition",
        "y": "Hip Height (cm)",
        "hue": "Perturbation State",
        "hue_order": perturbation_state_order,
    }

    # Step Width Plot
    print("\nStep Width Stats\n")
    plt.title("Step Width between conditions")
    sw_plot = sns.barplot(**sw_plot_params, ci=95, capsize=0.05)
    plt.legend(loc="best", fontsize=16)
    annotator = Annotator(sw_plot, condition_pairs, **sw_plot_params)
    annotator.new_plot(sw_plot, condition_pairs, plot="barplot", **sw_plot_params)
    annotator.configure(
        hide_non_significant=True, test="t-test_ind", text_format="star", loc="inside"
    )
    annotator.apply_test().annotate(line_offset_to_group=0.2, line_offset=0.1)

    fig = mpl.pyplot.gcf()
    fig.set_size_inches(19.8, 10.80)

    if save_stepw is True:
        plt.savefig("./aging/stepw_all.pdf", dpi=300)
        con_sw_combo.to_csv("./stepw_aging.csv", index=False)
        print("\nStep Width Analysis saved")
        plt.show()
    else:
        print("\nStep Width Analysis Not Saved")
        plt.show()

    # Hip Height Plot
    print("\nHip Height Stats\n")
    plt.title("Hip Height Between Conditions")
    hiph_plot = sns.barplot(**hiph_plot_params, ci=95, capsize=0.05)
    plt.legend(loc="best", fontsize=16)
    annotator = Annotator(hiph_plot, condition_pairs, **hiph_plot_params)
    annotator.new_plot(hiph_plot, condition_pairs, plot="barplot", **hiph_plot_params)
    annotator.configure(
        hide_non_significant=True, test="t-test_ind", text_format="star", loc="inside"
    )

    annotator.apply_test().annotate(line_offset_to_group=0.2, line_offset=0.1)

    fig = mpl.pyplot.gcf()
    fig.set_size_inches(19.8, 10.80)
    fig.tight_layout()

    if save_hiph is True:
        hiph_df.to_csv("./hiph_aging.csv", index=False)
        plt.savefig("./aging/hiph_all.pdf", dpi=300)
        print("\nHip Height Plot saved")
        plt.show()
    else:
        print("\nHip Height Analysis Not Saved")
        plt.show()


if __name__ == "__main__":
    main()
