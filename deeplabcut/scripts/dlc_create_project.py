import os
import platform
from datetime import datetime
from pathlib import Path

import deeplabcut


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
projectName = "mole_rat_right"
experimenterName = "kenzie"
targetForProject = "/Users/kenzie_mackinnon/sync/lab-analysis/deeplabcut/"

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
    videoImportPath = str(input("Enter file path for videos you want to import: "))

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
