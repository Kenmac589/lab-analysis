# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # This is the analysis for Center of Mass Project
#
# The goal of this is notebook is to bring together the general guidance on using DeepLabCut, while making the analysis executable in context with the associated directions. The first step will be to activate the relevant `conda` environment, which contains DeepLabCut.
# - In the case of the GPU computer, this will be done by launching the *anaconda prompt (anaconda powershell prompt is also fine)* in administator mode and typing `conda activate dlc-windowsGPU-2023'`
# - If you are using you're own PC, then the command would be `conda activate DEEPLABCUT`.
#
#
#
# ## General Overview
#
# 1. Import relevant packages and create project
# 2. Extract frames from imported videos
# 3. Label frames from videos to denote anatomincal landmarks
# 4. Train Neural Network (GPU Intensive)
# 5. Evalualte Network

# %% [markdown]
# First step will be to import deeplabcut.
# - Running blocks of code in jupyter is done by making sure that you are one the block you want to run and either pressing the run button above or the shortcut (Ctrl+Enter).

# %%
import deeplabcut
import numpy as np
import pandas as pd

# %% [markdown]
# ### Project Creation
#
# Here we will use the block written below to create a new project.
#
# > Do not run the block below if you have already made your project. You can move down to the block containing `config_path`.

# %%
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 11:09:35 2023

@author: Kenzie MacKinnon

The purpose of this script is to facilitate the import and creation of a new
DeepLabCut project.

The does project creation in a more interactive way if you run it.
You can also assign the appropriate variable in the code block below

"""

# %% Imports
import os
import platform
from datetime import datetime
from pathlib import Path


# Functions
def filePathList(dir_path):
    file_paths = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


def pathconvUnixToDos(paths):
    dos_paths = []
    for path in paths:
        dos_path = path.replace("/", "\\")
        dos_paths.append(dos_path)
    return dos_paths


def directoryPresent(targetPath, projName):
    path = os.path.join(projName, targetPath)

    # Will return boolean
    return os.path.isdir(path)


# Gathering User input
projectName = "4mo_and_1yrSham-preDTX"
experimenterName = "kenzie"
targetForProject = "/Users/kenzie_mackinnon/sync/lab-analysis/deeplabcut/1yr/"

# Depending on the OS processing the file we will run different functions to create file path lists
operatingSystem = platform.system()

if operatingSystem == "Windows":
    print("Window operating system detected!")

    # Path where recorded videos can currently be found
    videoImportPath = "\\Kenzie\\CoM\\DTR\\DTR-M5\\DTR-M5-20230404_pre-DTX\\"

    # Creating list of file paths for each video in specified folder
    file_paths = filePathList(videoImportPath)

    # Changing file paths from Unix format to DOS for videos
    dos_path_conversion = pathconvUnixToDos(file_paths)
    file_paths = dos_path_conversion

    # Changing file paths from Unix format to DOS for target directory
    # unixPathComponents = targetForProject.split('/')
    # targetForProject = os.path.join(*unixPathComponents).replace('/', os.sep)
    # print(targetForProject)
elif operatingSystem == "Linux":
    print("Linux operating system detected!")
    # Path where recorded videos can currently be found
    videoImportPath = str(input("Enter file path for videos you want to import: "))

    # Creating list of file paths for each video in specified folder
    file_paths = filePathList(videoImportPath)
elif operatingSystem == "Darwin":
    print("Darwin(MacOS) operating system detected!")
    # Path where recorded videos can currently be found
    videoImportPath = str("~/sync/lab-analysis/deeplabcut/1yr/4mo_and_1yrSham-preDTX")

    # Creating list of file paths for each video in specified folder
    file_paths = filePathList(videoImportPath)
else:
    print("Operating system not detected!")
    print("Falling back onto Unix path protocol")
    # Path where recorded videos can currently be found
    videoImportPath = str(input("Enter file path for videos you want to import: "))

    # Creating list of file paths for each video in specified folder
    file_paths = filePathList(videoImportPath)

# Checking result of variable file path importing
print("Project Name: " + projectName)
print("Experimenter: " + experimenterName)
print("Output of file paths:")
print("--------------------")
print(file_paths)
# %% Creation of project

# Checking to see if project with same name already exists
current_date = datetime.now().strftime("%Y-%m-%d")

newProjectName = projectName + "-" + experimenterName + "-" + current_date

if directoryPresent(newProjectName, targetForProject):
    print(f"Directory {newProjectName} already exists in {targetForProject}.")
else:
    print(f"Directory {newProjectName} does not exist in {targetForProject}.")
    config_path = deeplabcut.create_new_project(
        projectName,
        experimenterName,
        file_paths,
        working_directory=(targetForProject),
        copy_videos=True,
    )

# # %% Extract frames from videos
# deeplabcut.extract_frames(config_path, mode='automatic', userfeedback=False)
#
# # %% Label frames
# deeplabcut.label_frames(config_path)
#
# # %% Check Annotated Frames
# deeplabcut.check_labels(config_path, visualizeindividuals=True)
#

# %% [markdown]
# What you need to add to the bottom of config file
#
# ```yaml
# bodyparts:
# - calib_1
# - calib_2
# - calib_3
# - calib_4
# - calib_5
# - calib_6
# - iliac_crest
# - hip
# - knee
# - ankle
# - metatarsal
# - toe
# - fl_toe
# - mirror_lhl
# - mirror_rhl
# - mirror_lfl
# - mirror_rfl
# - mirror_com
# - mirror
# # The following two lines tell the clustering algorithm from where to where (as a fraction of the video length) frames have to be extracted when using deeplabcut.extract_frames()
#
# # Fraction of video to start/stop when extracting frames for labeling/refinement
#
#     # Fraction of video to start/stop when extracting frames for labeling/refinement
# start: 0
# stop: 1
# numframes2pick: 20
#
#     # Plotting configuration
# skeleton:
# - - iliac_crest
#   - hip
# - - hip
#   - knee
# - - knee
#   - ankle
# - - ankle
#   - metatarsal
# - - metatarsal
#   - toe
# skeleton_color: black
# pcutoff: 0.6
# dotsize: 8
# alphavalue: 0.7
# colormap: rainbow
#
#     # Training,Evaluation and Analysis configuration
# TrainingFraction:
# - 0.95
# iteration: 0
# default_net_type: resnet_50
# default_augmenter: default
# snapshotindex: -1
# batch_size: 8
#
#     # Cropping Parameters (for analysis and outlier frame detection)
# cropping: false
#     #if cropping is true for analysis, then set the values here:
# x1: 0
# x2: 640
# y1: 277
# y2: 624
#
#     # Refinement configuration (parameters from annotation dataset configuration also relevant in this stage)
# corner2move2:
# - 50
# - 50
# move2corner: true
#
# ```

# %% [markdown]
# On windows server config path

# %%
config_path = "/Users/kenzie_mackinnon/sync/lab-analysis/deeplabcut/1yr/1yrDTRnoRosa-preDTX-kenzie-2024-01-31/config.yaml"
videofile_path = "/Users/kenzie_mackinnon/sync/lab-analysis/deeplabcut/1yr/1yrDTRnoRosa-preDTX-kenzie-2024-01-31/videos/"
VideoType = "avi"

# %% [markdown]
# On unix server config path

# %%
config_path = "/Users/kenzie_mackinnon/sync/lab-analysis/deeplabcut/dlc-dtr/DTR-M5/test_output-kenzie-2024-01-04/config.yaml"

# %%
deeplabcut.extract_frames(
    config_path, mode="automatic", algo="kmeans", userfeedback=False, crop=True
)

# %%
deeplabcut.label_frames(config_path)

# %% [markdown]
# ### Checking Labeling

# %%
deeplabcut.check_labels(config_path, visualizeindividuals=False)

# %% [markdown]
# ### Create Training Dataset
#
# Only run this step where you are going to train the network. If you label on your laptop but move your project folder to Google Colab or AWS, lab server, etc, then run the step below on that platform! If you labeled on a Windows machine but train on Linux, this is fine as of 2.0.4 onwards it will be done automatically (it saves file sets as both Linux and Windows for you).

# %%
deeplabcut.create_training_dataset(config_path, augmenter_type="imgaug")

# %% [markdown]
# ### Network Training
#
# This part is where you would want to be leveraging the GPU's on the big PC to training the Neural Network.

# %%
deeplabcut.train_network(config_path)

# %%
# %matplotlib
# Network evaluation

deeplabcut.evaluate_network(config_path, plotting=True)

# %% [markdown]
# # Begin Video Analysis

# %%
deeplabcut.analyze_videos(config_path, videofile_path, videotype=VideoType)

# %% [markdown]
# # Plot the Trajectory

# %%
deeplabcut.plot_trajectories(config_path, videofile_path, videotype=VideoType)

# %%
deeplabcut.create_labeled_video(config_path, videofile_path, videotype=VideoType)

# %% [markdown]
# # Finding frames with abnormal body part Distances
#
#

# %%

max_dist = 100
df = pd.read_hdf("path_to_your_labeled_data_file")
bpt1 = df.xs("head", level="bodyparts", axis=1).to_numpy()
bpt2 = df.xs("tail", level="bodyparts", axis=1).to_numpy()
# We calculate the vectors from a point to the other
# and group them per frame and per animal.
try:
    diff = (bpt1 - bpt2).reshape((len(df), -1, 2))
except ValueError:
    diff = (bpt1 - bpt2).reshape((len(df), -1, 3))
dist = np.linalg.norm(diff, axis=2)
mask = np.any(dist >= max_dist, axis=1)
flagged_frames = df.iloc[mask].index
