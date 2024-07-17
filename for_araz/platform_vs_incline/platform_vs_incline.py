import dlc2kinematics as dlck
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from dlc2kinematics import Visualizer2D
from kinsynpy import dlctools as dt
from kinsynpy import latstability as ls

hip_heights = np.array([])
for i in range(14):
    if i < 10:
        video = f"{i}"
        df, bodyparts, scorer = dlck.load_data(
            f"./incline_recordings/2024-05-09_00000{video}DLC_resnet50_SN-7 Post SNMay11shuffle1_500000.h5"
        )

        # # NOTE: Very important this is checked before running
        # mouse_number = 2
        # manual_analysis = False
        # save_auto = False
        filter_k = 13
        #
        # # Settings before running initial workup from DeepLabCut
        # figure_title = f"Step Cycles for level-test-M{mouse_number}-vid-{video}"
        # figure_filename = f"./treadmill_level_test/emg-test-2-dropped/level_mos_analysis/m{mouse_number}-{video}.svg"
        # step_cycles_filename = f"./treadmill_level_test/emg-test-2-dropped/level_mos_analysis/m{mouse_number}-step-cycles-{video}.csv"
        #
        # # Some things to set for plotting/saving
        # lmos_filename = f"./treadmill_level_test/emg-test-2-dropped/level_mos_analysis/m{mouse_number}-lmos-{video}.csv"
        # rmos_filename = f"./treadmill_level_test/emg-test-2-dropped/level_mos_analysis/m{mouse_number}-rmos-{video}.csv"
        # mos_figure_title = (
        #     f"Measurement of Stability For Level Test M{mouse_number}-{video}"
        # )
        # mos_figure_filename = f"./treadmill_level_test/emg-test-2-dropped/level_mos_analysis/m{mouse_number}-mos-{video}.svg"
        calib_markers = [
            "calib_1",
            "calib_2",
            "calib_3",
            "calib_4",
            "calib_5",
            "calib_6",
        ]

        # For visualizing skeleton
        # config_path = (
        #     "../../deeplabcut/dlc-dtr/dtr_update_predtx-kenzie-2024-04-08/config.yaml"
        # )
        # foi = f"../../deeplabcut/dlc-dtr/DTR-M6/DTR-M6-preDTX/DTR-M6-preDTX_0000{video}DLC_resnet50_dtr_update_predtxApr8shuffle1_1110000_filtered.h5"
        # viz = Visualizer2D(config_path, foi, form_skeleton=True)
        # viz.view(show_axes=True, show_grid=True, show_labels=True)
        # plt.show()

        calib_factor = dt.dlc_calibrate(df, bodyparts, scorer, calib_markers)
        dt.xcom

        # Grabbing toe marker data
        toe = df[scorer]["toe"]
        hip = df[scorer]["hip"]

        # Converting to numpy array
        toe_np = pd.array(toe["x"])
        toe_np = toe_np / calib_factor
        toey_np = pd.array(toe["y"])
        toey_np = toey_np / calib_factor
        hipy_np = pd.array(hip["y"])
        hipy_np = hipy_np / calib_factor

        # comy_np = pd.array(com["y"])
        # comy_np = comy_np / calib_factor
        time = np.arange(0, len(toey_np), 1)
        # time_vec = np.vectorize(frame_to_time)
        # time = time_vec(time)
        time = dt.frame_to_time(time)

        # Filtering to clean up traces like you would in spike
        toe_smooth = dt.median_filter(toe_np, filter_k)
        toe_smooth = sp.signal.savgol_filter(toe_smooth, 20, 3)
        # com_med = median_filter(comy_np, filter_k)

        hip_h = dt.hip_height(toey_np, hipy_np)

        hip_heights = np.append(hip_heights, hip_h)
    else:
        video = f"{i}"
        df, bodyparts, scorer = dlck.load_data(
            f"./incline_recordings/2024-05-09_0000{video}DLC_resnet50_SN-7 Post SNMay11shuffle1_500000.h5"
        )

        # # NOTE: Very important this is checked before running
        # mouse_number = 2
        # manual_analysis = False
        # save_auto = False
        filter_k = 13
        #
        # # Settings before running initial workup from DeepLabCut
        # figure_title = f"Step Cycles for level-test-M{mouse_number}-vid-{video}"
        # figure_filename = f"./treadmill_level_test/emg-test-2-dropped/level_mos_analysis/m{mouse_number}-{video}.svg"
        # step_cycles_filename = f"./treadmill_level_test/emg-test-2-dropped/level_mos_analysis/m{mouse_number}-step-cycles-{video}.csv"
        #
        # # Some things to set for plotting/saving
        # lmos_filename = f"./treadmill_level_test/emg-test-2-dropped/level_mos_analysis/m{mouse_number}-lmos-{video}.csv"
        # rmos_filename = f"./treadmill_level_test/emg-test-2-dropped/level_mos_analysis/m{mouse_number}-rmos-{video}.csv"
        # mos_figure_title = (
        #     f"Measurement of Stability For Level Test M{mouse_number}-{video}"
        # )
        # mos_figure_filename = f"./treadmill_level_test/emg-test-2-dropped/level_mos_analysis/m{mouse_number}-mos-{video}.svg"
        calib_markers = [
            "calib_1",
            "calib_2",
            "calib_3",
            "calib_4",
            "calib_5",
            "calib_6",
        ]

        # For visualizing skeleton
        # config_path = (
        #     "../../deeplabcut/dlc-dtr/dtr_update_predtx-kenzie-2024-04-08/config.yaml"
        # )
        # foi = f"../../deeplabcut/dlc-dtr/DTR-M6/DTR-M6-preDTX/DTR-M6-preDTX_0000{video}DLC_resnet50_dtr_update_predtxApr8shuffle1_1110000_filtered.h5"
        # viz = Visualizer2D(config_path, foi, form_skeleton=True)
        # viz.view(show_axes=True, show_grid=True, show_labels=True)
        # plt.show()

        calib_factor = dt.dlc_calibrate(df, bodyparts, scorer, calib_markers)
        dt.xcom

        # Grabbing toe marker data
        toe = df[scorer]["toe"]
        hip = df[scorer]["hip"]

        # Converting to numpy array
        toe_np = pd.array(toe["x"])
        toe_np = toe_np / calib_factor
        toey_np = pd.array(toe["y"])
        toey_np = toey_np / calib_factor
        hipy_np = pd.array(hip["y"])
        hipy_np = hipy_np / calib_factor

        # comy_np = pd.array(com["y"])
        # comy_np = comy_np / calib_factor
        time = np.arange(0, len(toey_np), 1)
        # time_vec = np.vectorize(frame_to_time)
        # time = time_vec(time)
        time = dt.frame_to_time(time)

        # Filtering to clean up traces like you would in spike
        toe_smooth = dt.median_filter(toe_np, filter_k)
        toe_smooth = sp.signal.savgol_filter(toe_smooth, 20, 3)
        # com_med = median_filter(comy_np, filter_k)

        hip_h = dt.hip_height(toey_np, hipy_np)

        hip_heights = np.append(hip_heights, hip_h)


print(np.mean(hip_heights))
# legend = ["hip_y", "toe_y"]
# plt.plot(time, hipy_np)
# plt.plot(time, toey_np)
# plt.legend(legend)
# plt.tight_layout()
# plt.show()
