import dlc2kinematics as dlck
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from kinsynpy import dlctools as dt

# Loading in a dataset
video = "03"
df, bodyparts, scorer = dlck.load_data(
    f"../../deeplabcut/dlc-dtr/DTR-M6/DTR-M6-preDTX/DTR-M6-preDTX_0000{video}DLC_resnet50_dtr_update_predtxApr8shuffle1_1110000_filtered.h5"
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

# Grabbing toe marker data
toe = df[scorer]["toe"]
hip = df[scorer]["hip"]
lhl = df[scorer]["mirror_lhl"]
rhl = df[scorer]["mirror_rhl"]
lfl = df[scorer]["mirror_lfl"]
rfl = df[scorer]["mirror_rfl"]
com = df[scorer]["mirror_com"]

# Converting to numpy array
toe_np = pd.array(toe["x"])
toe_np = toe_np / calib_factor
toey_np = pd.array(toe["y"])
toey_np = toey_np / calib_factor
hipy_np = pd.array(hip["y"])
hipy_np = hipy_np / calib_factor

comy_np = pd.array(com["y"])
comy_np = comy_np / calib_factor
time = np.arange(0, len(comy_np), 1)
# time_vec = np.vectorize(frame_to_time)
# time = time_vec(time)
time = dt.frame_to_time(time)
rfl_np = pd.array(rfl["y"])
rfl_np = rfl_np / calib_factor
rhl_np = pd.array(rhl["y"])
rhl_np = rhl_np / calib_factor
lfl_np = pd.array(lfl["y"])
lfl_np = lfl_np / calib_factor
lhl_np = pd.array(lhl["y"])
lhl_np = lhl_np / calib_factor

# Filtering to clean up traces like you would in spike
toe_smooth = dt.median_filter(toe_np, filter_k)
toe_smooth = sp.signal.savgol_filter(toe_smooth, 20, 3)
# com_med = median_filter(comy_np, filter_k)
com_med = sp.signal.savgol_filter(comy_np, 40, 3)

rfl_med = dt.median_filter(rfl_np, filter_k)
rfl_med = sp.signal.savgol_filter(rfl_med, 30, 3)
rhl_med = dt.median_filter(rhl_np, filter_k)
rhl_med = sp.signal.savgol_filter(rhl_med, 30, 3)
lfl_med = dt.median_filter(lfl_np, filter_k)
lfl_med = sp.signal.savgol_filter(lfl_med, 30, 3)
lhl_med = dt.median_filter(lhl_np, filter_k)
lhl_med = sp.signal.savgol_filter(lhl_med, 30, 3)

com_slope = dt.spike_slope(com_med, 30)
hip_h = dt.hip_height(toey_np, hipy_np)
xcom_test = dt.xcom(comy=com_med, vcom=com_slope, hip_height=hip_h)

legend = ["Same as DLC into spike", "Filtered CoM", "xCoM"]
plt.plot(time, comy_np)
plt.plot(time, com_med)
plt.plot(time, xcom_test)
plt.tight_layout()
plt.legend(legend)
plt.show()
