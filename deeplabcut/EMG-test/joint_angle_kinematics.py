import dlc2kinematics as dlck
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# from dlc2kinematics import Visualizer2D
from kinsynpy import dlctools as dlt


def main():

    # NOTE: At this time I am using videos from:
    # - M1-pre-emg: vids 3, 4
    # - M2-pre-emg: vids 0, 1, 2
    # - M1-post-emg: vids 0, 1, 2, 3
    # - M2-post-emg: vids 1, 8, 9

    mouse = 1
    video = "03"
    select_region = False

    df, bodyparts, scorer = dlck.load_data(
        f"./EMG-test-1/EMG-test-1-pre-emg/EMG-test-1-pre-emg_0000{video}DLC_resnet50_dtr_update_predtxApr8shuffle1_1110000_filtered.h5"
    )

    # For visualizing skeleton
    config_path = "../../deeplabcut/supercom-kenzie-2024-08-13/config.yaml"
    foi = "./EMG-test-1/EMG-test-1-pre-emg/EMG-test-1-pre-emg_000003DLC_resnet50_dtr_update_predtxApr8shuffle1_1110000_filtered_skeleton.csv"

    # Need Hip, Knee, and Ankle
    joints_dict = {}
    joints_dict["hip"] = ["iliac_crest", "hip", "knee"]
    joints_dict["knee"] = ["hip", "knee", "ankle"]
    joints_dict["ankle"] = ["knee", "ankle", "metatarsal"]

    joint_angles = dlck.compute_joint_angles(
        df,
        joints_dict=joints_dict,
        save=True,
        destfolder="./joint_angles/",
        dropnan=True,
        smooth=True,
        filter_window=40,
        order=3,
    )

    hip_angle = joint_angles["hip"]
    knee_angle = joint_angles["knee"]
    ankle_angle = joint_angles["ankle"]

    # Loading in skeleton
    sk_df = pd.read_csv(foi)
    limb_names = [
        "iliac_crest_hip",
        "hip_knee",
        "knee_ankle",
        "ankle_metatarsal",
        "metatarsal_toe",
    ]

    calib_markers = [
        "calib_1",
        "calib_2",
        "calib_3",
        "calib_4",
        "calib_5",
        "calib_6",
    ]

    calib_factor = dlt.dlc_calibrate(df, bodyparts, scorer, calib_markers)
    limb_diffs = dlt.limb_measurements(
        sk_df,
        limb_names,
        calib_factor,
        save_as_csv=True,
        csv_filename=f"./joint_angles/emg-test-joint_angle-m{mouse}-vid-{video}.csv",
    )
    print(f"List of bodyparts:\n{bodyparts}\n")
    print(f"List of joints:\n{joint_angles}\n")
    print(f"Length of limb coordinates in cm:\n{limb_diffs}")

    # Grabbing marker data
    toex_np = dlt.mark_process(df, scorer, "toe", "x", calib_factor)
    toey_np = dlt.mark_process(df, scorer, "toe", "y", calib_factor)
    hipy_np = dlt.mark_process(df, scorer, "hip", "y", calib_factor)
    comy_np = dlt.mark_process(df, scorer, "mirror_com", "y", calib_factor)
    rfly_np = dlt.mark_process(df, scorer, "mirror_rfl", "y", calib_factor)
    rhly_np = dlt.mark_process(df, scorer, "mirror_rhl", "y", calib_factor)
    lfly_np = dlt.mark_process(df, scorer, "mirror_lfl", "y", calib_factor)
    lhly_np = dlt.mark_process(df, scorer, "mirror_lhl", "y", calib_factor)
    rflx_np = dlt.mark_process(df, scorer, "mirror_rfl", "x", calib_factor)
    rhlx_np = dlt.mark_process(df, scorer, "mirror_rhl", "x", calib_factor)
    lflx_np = dlt.mark_process(df, scorer, "mirror_lfl", "x", calib_factor)
    lhlx_np = dlt.mark_process(df, scorer, "mirror_lhl", "x", calib_factor)
    miry_np = dlt.mark_process(df, scorer, "mirror", "y", calib_factor, smooth_wind=70)

    print(f"toex {len(toex_np)}")
    print(f"hip_angle {len(hip_angle)}")

    np.savetxt("./hip_angle_test.csv", hip_angle, delimiter=",")
    # Selecting a Given Region
    if select_region is True:
        reg_start, reg_stop = dlt.analysis_reg_sel(mirror_y=miry_np, com_y=comy_np)
        toex_np = toex_np[reg_start:reg_stop]
        toey_np = toey_np[reg_start:reg_stop]
        hipy_np = hipy_np[reg_start:reg_stop]
        comy_np = comy_np[reg_start:reg_stop]
        rfly_np = rfly_np[reg_start:reg_stop]
        rhly_np = rhly_np[reg_start:reg_stop]
        lfly_np = lfly_np[reg_start:reg_stop]
        lhly_np = lhly_np[reg_start:reg_stop]
        rflx_np = rflx_np[reg_start:reg_stop]
        rhlx_np = rhlx_np[reg_start:reg_stop]
        lflx_np = lflx_np[reg_start:reg_stop]
        lhlx_np = lhlx_np[reg_start:reg_stop]
    else:
        print("Looking at entire recording")

    plt.plot(hip_angle)
    plt.show()


if __name__ == "__main__":
    main()
