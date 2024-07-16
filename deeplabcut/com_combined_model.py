import deeplabcut as dlc

project_name = "com_combined_model"
experimenter = "kenzie"
project_path = "/Users/kenzie_mackinnon/sync/lab-analysis/deeplabcut/"
video_path = "/Users/kenzie_mackinnon/sync/lab-analysis/deeplabcut/dlc-dtr/analysis_selected_videos"


config_path = dlc.create_new_project(
    project_name,
    experimenter,
    [video_path],
    project_path,
    copy_videos=True,
    videotype="avi",
)
