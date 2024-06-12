# Notes from Kinematic analysis

- This for notes regarding what I am noticing while doing kinematic analysis.

# Hip Height

[hip_heights](hip_heights.md)

# Measurement of Dynamic Lateral Stability (MoS)

NOTE: Really important information on certain data files that are problematic.
- Review:
    - wt-2-non
    - wt-1-per
    - wt-2-per
    - wt-2-sin
    - egr3-7-non
    - egr3-7-per
    - egr3-8-per
- General trends seem to be the same


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

### Egr3-M9

- Added a second perturbation recording under cursors 7 to 8
    - related export is the first of the perturbation ones
    - other is between 3-4
- After reviewing plots of the files, for the sinusoidal trial
    - File 1 `./egr3_data/egr3-9-sinus-xcom-redo-1.txt` is from cursor **5-6**
    - File 2 `./egr3_data/egr3-9-sinus-xcom-redo-2.txt` is from cursor **9-0**

### Egr3-10

- For the perturbation trial, the file with `pt1` is from cursor 3-7 and `pt2` is from 7-4. However, 
    - NOTE: That cursor 7 was moved to filter out region of inactivity.
    - And the *full* file is just from 3-4

# DTR Mice

NOTE: For the sw on and offset channels are only cleaned for conditions I will be using. Don't just go ahead and think they are cleaned and can go to the server.

## DTR-M1

- Not really useable as there is nothing for any perturbation for the preDTX.

## DTR-M2

### DTR-M2-PreDTX

- 
 
- Even non-perturbation for DTR-M2 is shit but kind of salvageable.

### DTR-M2-Post-DTX

**Videos used:** 2, 13, 19
NOTE: Post-DTX mice seem to be kicking out their feet quite a bit regardless of normal locomotion

- No real signs for concern regarding the stomach touching the ground.
- Trials selected for postdtx analysis (starting at 0)
    - non-perturbation: video 2 
    - perturbation: video 13
        - movement is a little spastic with stomach drop maybe once but not chronic
    - sinusoidal: video 19
        - 

## DTR-M3


### DTR-M3-Post-DTX

**Videos used:** 0, 9, 14
- Trials selected for postdtx analysis (starting at 0)
    - non-perturbation: video 0 
        - [ ] Review with Turgay
    - perturbation: video 9
    - sinusoidal: video 14

## DTR-M5

### DTR-M5-Post-DTX

**Videos used:** 0, 7, 14
- Trials selected for postdtx analysis (starting at 0)
    - non-perturbation: video 0 
    - perturbation: video 7
        - I split it up into 2 portions as there is a pause in the middle of the recording.
        - It is otherwise the most clean I've found.
        - Definitely all cautious walking.
    - sinusoidal: video 14

## DTR-M6-8

NOTE: Still definitely noticing some of the drift I mentioned previously with the com leaning towards the left side.

## Kinematics for DTR M6-M8

- M6-preDTX-000019 video is fairly poor, so keep that in mind in terms of the model performance.
    
- PreDTX
    - M6:
        - Length of thigh: 1.5
        - Length of shank: 1.5
    - M7:
        - Length of thigh: 1.2
        - Length of shank: 1.4
- Post-DTX
    - M6:
        - Length of thigh: 1.4
        - Length of shank: 1.6
    - M7:
        - Length of thigh: 1.4
        - Length of shank: 1.6
    - M8:
        - Length of thigh: 1.4
        - Length of shank: 1.6

- There are really no valid recordings for perturbation or sinusoidal conditions.

## DTR-M6

- Both sets of recordings seem to be quite usable and clear.

### DTR-M6 Pre-DTX


### DTR-M6 Post-DTX

- I used videos 3, 14, 18


## DTR-M7

### DTR-M7 Pre-DTX

- For the perturbation recording, I am going with the auto data.

### DTR-M7 Post-DTX

- There is nothing really present for M7 Post-DTX, mouse was severely ataxic.



## DTR-M8

### DTR-M8 Pre-DTX

- There is no kinematic recordings done a 0.100 m/s for random perturbation.
- There is for sinusoidal at least.



### DTR-M8 Post-DTX

- Stomach seems to be dragging quite a bit in videos overall
- Yeah confirming this again on 27/05/2024 and yeah most recordings aren't really usable.
    - I checked video 0, 1, 3,  

