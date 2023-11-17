
# Still in Spike

- Aim for between 15-20 step cycles

1. DC Remove -> remove zero
2. Rectify
3. Smoothen

When creating channels like sw onset
- Create new memory buffer
- Take ToeX and add the process of slope
- Import toex channel with every time it rises through

- Also for cleaning up random blips on DLC, you the channel process `median filter`

[spike_configuration](spike_configuration.md)

# Channels Used for CoM Calculations

- Center of Mass(CoMx) = Ch36
- Hindlimb X(HRx) = Ch29
- Forelimb Y(FRx) = Ch34
- 24 toey (cm)

# Video Analysis log

- Working on what appears to be 3rd recording from 373.589-381.919 second range.
  - Refined parameters to include 15 steps worth 373.73715-377.34641 (0-3.60926s)
- Video is taken at .200 m/s

# Regarding R scripts from @Santuz2019

- Data given:
    - raw EMG from two trials


# CoM-M3-WT-20220420

25 recordings

- Working on what appears to be 3rd recording from 373.589-381.919 second range.
  - Refined parameters to include 15 steps worth 373.73715-377.34641 (0-3.60926s)
- Video is taken at .200 m/s

- Chosen that the cutoff for the floor of the treadmill to be 7.7733. This must be taken off of the HRy original values.

# CoM-M1-WT-20220422

27 recordings

# CoM-M2-WT-20220419

12 recordings

# Step Cycle Presentation

- Present by having 
  
# For DTR Mice Series (6 month group)

**The 6 month DTR mice have some channels that are mixed around from the order of implantation in the 1 yr mice which is based on the spike configuration currently being used**
- For Gastrocnemius (Gs)
    - 6 month: muscle 5, channel 8 in spike
    - 1 year: muscle 7, channel 10 in spike
- For Semitendinosus (St)
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

[recording_summary](./com/dtr/dtr-6-months/recording_summary.csv)

- [X] M1-pre
- [X] M1-post
- [X] M2-pre
- [ ] M2-post
    - [ ] Review data quality with Turgay.
- [ ] M3-pre
- [ ] M3-post
- [X] M5-pre
- [ ] M5-post

# First Committee Meeting Analysis

- Looking at 6m-M5-preDTX
    - First sinusoidal at 0.100 is broken into parts of constant stimulation and a full recording as well
    - Second sinusoidal at 0.400 much better looking in terms of just being cleaner
