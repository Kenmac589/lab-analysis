import deeplabcut as dlc

# Setting things that vary
config_path = "/Users/kenzie_mackinnon/sync/lab-analysis/deeplabcut/dlc-dtr/dtr_update_predtx-kenzie-2024-04-08/config.yaml"
video_folder = "/Users/kenzie_mackinnon/sync/lab-analysis/deeplabcut/dlc-dtr/dtr_update_predtx-kenzie-2024-04-08/videos"


# Analyze Videos
dlc.analyze_videos(config_path, [video_folder], videotype="avi", save_as_csv=True)

# Filter predictions
dlc.filterpredictions(config_path, [video_folder], filtertype="median")

# Plot trajectories
dlc.plot_trajectories(config_path, [video_folder], videotype="avi", filtered=True)

# Create Labelled videos
dlc.create_labeled_video(config_path, [video_folder], videotype="avi", filtered=True)
