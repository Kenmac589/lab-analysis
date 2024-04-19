# Data Analysis Notes

## Observational Notes

[emg_analysis](emg_analysis.md)
[kinematic_analysis](kinematic_analysis.md)

## Topics

[anaylsis_background](anaylsis_background.md)
[spike_guidance](spike_guidance.md)
[spike_configuration](spike_configuration.md)
[com_project](com_project.md)
[kinematic_results](../kinematics/cycle_analysis/kinematic_results.md)


# Wild-type Controls

- Most of what's being looked at here is 0.100 m/s

## WT 1

    - For non-perturbation @0.100, I used first recording
    - synergies to use: 3
    - For perturbation @0.100 I used recording starting at ~1067
    - synergies to use: 3

## WT 4

- I used first non-perturbation recording and the perturbation recording starts at around 1205

# For DTR Mice Series (4 month group)

**The 6 month DTR mice have some channels that are mixed around from the order of implantation in the 1 yr mice which is based on the spike configuration currently being used**
- For Gastrocnemius (Gs)
    - CoM: muscle 7, channel 10 in spike
    - 6 month: muscle 5, channel 8 in spike
    - 1 year: muscle 7, channel 10 in spike
- For Semitendinosus (St)
    - CoM: muscle 5, channel 8 in spike
    - 6 month: muscle 7, channel 10 in spike
    - 1 year: muscle 5, channel 8 in spike


Here are the values in cm (first thigh and then shank):

# For pre DTX recordings:

M1: 1.5 -1.5
M2: 1.6 - 1.8
M3: 1.4 - 1.6
M5: 1.5 - 1.8

For post DTX recordings:

M1: 1.4 -1.6
M2: 1.5 - 1.6
M3: 1.4 - 1.7
M5: 1.4 - 1.7

## Working Through Files

[recording_summary](../com/dtr/dtr-6-months/recording_summary.csv)

**Turgay is planning on redoing all the 4 month group**

- [X] M5-pre
- [X] M5-post

# First Committee Meeting Analysis

- Looking at 6m-M5-preDTX
    - First sinusoidal at 0.100 is broken into parts of constant stimulation and a full recording as well
    - Second sinusoidal at 0.400 much better looking in terms of just being cleaner
# Post Meeting Analysis

- Working through the wild type mice

# 1yr group

## 1yrDTRnoRosa-preDTX Analysis

- for M1 I labelled
    - vid 0
    - vid 08
    - vid 22
- for M2 I labelled
    - vid 0
    - vid 9
    - vid 21
- 0.100 m/s videos from recordings
    - M1-oct19-2023: 0 (non-per), 7-16 (per), 17-20 (sin)
    - M2-oct23-2023: 0 (non-per), 8-17, (per), 17-20 (sin)

```python
videos_of_interest = [
    '/Users/kenzie_mackinnon/sync/lab-analysis/deeplabcut/1yr/1yrDTRnoRosa-preDTX-kenzie-2024-01-31_analyzed/videos/1yrDTRnoRosa-M1-19102023_000000.avi',
    '/Users/kenzie_mackinnon/sync/lab-analysis/deeplabcut/1yr/1yrDTRnoRosa-preDTX-kenzie-2024-01-31_analyzed/videos/1yrDTRnoRosa-M1-19102023_000007.avi',
    '/Users/kenzie_mackinnon/sync/lab-analysis/deeplabcut/1yr/1yrDTRnoRosa-preDTX-kenzie-2024-01-31_analyzed/videos/1yrDTRnoRosa-M1-19102023_000008.avi',
    '/Users/kenzie_mackinnon/sync/lab-analysis/deeplabcut/1yr/1yrDTRnoRosa-preDTX-kenzie-2024-01-31_analyzed/videos/1yrDTRnoRosa-M1-19102023_000018.avi',
    

]

```

# Measurement of Dynamic Lateral Stability (MoS)

## Egr3 and WT sets

- Mostly already done, however, I'm using extra channels to clean some stuff up.

- Going over the possibly affected data files
    - So far there is
    - By the looks of it, all of them with perturbation and sinusoidal
        - [X] Perturbation and sinus for egr 10
            - Noting that for the perturbation, despite overlap, it's during a stimulation, mouse must of stumbled.
        - [X] Perturbation and sinus for egr 9
            - Added a second perturbation recording under cursors 7 to 8
                - related export is the first of the perturbation ones
                - other is between 3-4
        - [X] Perturbation and sinus for egr 8
        - [X] Perturbation and sinus for egr 7
        - [X] Perturbation and sinus for egr 6
        - [X] Definitely check all the WT's
            - [X] WT 1
            - [X] WT 2
            - [X] WT 3
            - [X] WT 4
            - [X] WT 5

- Seeing some weird stuff with the Egr3-7 mouse check videos
- NOTE: WT M2 and Egr3 M7 are kind of looking sketchy imo.

## For the DTR set

NOTE: For the sw on and offset channels are only cleaned for conditions I will be using. Don't just go ahead and think they are cleaned and can go to the server.

### DTR-M1

- Not really useable as there is nothing for any perturbation for the preDTX.

### DTR-M2

- Even non-perturbation for DTR-M2 is shit but kind of salvageable.

### DTR-

### Kinematics for DTR M6-M8

- M6-preDTX-000019 video is fairly poor, so keep that in mind in terms of the model performance.

