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

# DTR Mice

NOTE: For the sw on and offset channels are only cleaned for conditions I will be using. Don't just go ahead and think they are cleaned and can go to the server.

## DTR-M1

- Not really useable as there is nothing for any perturbation for the preDTX.

## DTR-M2

### DTR-M2-PreDTX

- 
 
- Even non-perturbation for DTR-M2 is shit but kind of salvageable.

### DTR-M2-PostDTX

NOTE: Post-DTX mice seem to be kicking out their feet quite a bit regardless of normal locomotion

- No real signs for concern regarding the stomach touching the ground.
- Trials selected for postdtx analysis (starting at 0)
    - non-perturbation: video 2 
    - perturbation: video 13
        - movement is a little spastic with stomach drop maybe once but not chronic
    - sinusoidal: video 19
        - 

## DTR-M3


### DTR-M3-PostDTX

- Trials selected for postdtx analysis (starting at 0)
    - non-perturbation: video 0 
        - [ ] Review with Turgay
    - perturbation: video 9
    - sinusoidal: video 14

## DTR-M5

### DTR-M5-PostDTX

- Trials selected for postdtx analysis (starting at 0)
    - non-perturbation: video 0 
    - perturbation: video 7
        - I split it up into 2 portions as there is a pause in the middle of the recording.
        - It is otherwise the most clean I've found.
        - Definitely all cautious walking.
    - sinusoidal: video 14

## DTR-M6-8

## Kinematics for DTR M6-M8

- M6-preDTX-000019 video is fairly poor, so keep that in mind in terms of the model performance.
    
- PreDTX
    - M6:
        - Length of thigh: 1.5
        - Length of shank: 1.5
    - M7:
        - Length of thigh: 1.2
        - Length of shank: 1.4
- PostDTX
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



## DTR-M8

- Stomach seems to be dragging quite a bit in videos overall
