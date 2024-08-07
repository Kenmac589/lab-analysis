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
            - Noting that for the perturbation, despite overlap, it's during a stimulation, mouse must have stumbled.
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

### Egr3-M6

- Videos used: 0, 10, 20

### Egr3-M7

- Videos used: 0, 9, 16


### Egr3-M8

- Videos used: 0, 18, 20


### Egr3-M9

- Videos used: 0, 15, 17, 21, 22

- Added a second perturbation recording under cursors 7 to 8
    - related export is the first of the perturbation ones
    - other is between 3-4
- After reviewing plots of the files, for the sinusoidal trial
    - File 1 `./egr3_data/egr3-9-sinus-xcom-redo-1.txt` is from cursor **5-6**
    - File 2 `./egr3_data/egr3-9-sinus-xcom-redo-2.txt` is from cursor **9-0**

### Egr3-10

- Videos Used: 0, 15, 27
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

### DTR-M5-Pre-DTX

- There are two perturbation recordings at cursors 3-4, 5-6

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
- Confirming this again on 27/05/2024 and yeah most recordings aren't really usable.
    - I checked video 0, 1, 3,  

# 1yr/1-6yr Mice

## 1yrDTRnoRosa M1

### 1yrDTRnoRosa M1 preDTX


- 0.100 non --> 0
    - There are more than one recording session revisit to see if other viable recordings
- 0.100 per --> 7 to 16
    - vid 07 kinda noisy at beginning but overall not too bad.
    - vid 08 similar, did save.
    - vid 09 more solid would use if non other look good
        - Hip height is kind of wild would take a look at.
    - vid 10 also pretty solid in terms of pretty tight measurements
    - vid 11 doesn't look too noisy would also include
    - vid 12 not super clean but not too bad
    - vid 13 only one real blip but solid otherwise
    - vid 14 there is likely one massive neg, this is miss labeling.
    - vid 15 like it worth review
    - vid 16 should investigate where perturbation is
- 0.100 sin --> 17 to 20
    - very clear which regions are the ones select
    - vid 17, 18 and 20 very solid, 19 not as much

- Currently going with vid 0, 9, 18 for plot


### 1yrDTRnoRosa M1 postDTX
 
- 0.100 non --> 0
- 0.100 per --> 3 to 12
- 0.100 sin --> 13 to 15

## 1yrDTRnoRosa M2 preDTX

- MoS values pretty level for this mouse, worth looking into
- Not so much the case for sinusoidal but definitely early on.

- 0.100 non --> 0 and 1
    - vid 0 look pretty good *noting that it's pretty level*
    - vid 1 is all over the place did not save
- 0.100 per --> 8 to 17
    - vid 8 also pretty level
    - vid 9 not bad, few blips but don't appear to interfere with chosen peaks.
    - vid 10 not too great
    - similar notes to 09 for 11
    - vid 12 not bad
    - vid 13 notable amount of sway but looks good
    - vid 14 little dodgy but not too bad
    - vid 15 little noisy but don't think it interfered
    - vid 16 looks pretty solid
    - vid 17 also doesn't look bad
- 0.100 sin --> 18 to 20
    - vid 18 good periods of both states
    - 19 and 20 also look good
- Currently using 0, 16, 19
    
## 1yrDTRnoRosa M2 postDTX

- 0.100 non --> 2 and 3
- 0.100 per --> 6 to 15
    - vid 11 is pretty solid
    - investigate vid 14 for potential fall
    - vid 17 definitely noisy but saved
    - vid 18 for sin definitely some lean
- 0.100 sin --> 16 to 18

# 1yrShamPvCre M1

- Current kinematics on server are from previous model consider training a new one.

## 1yrShamPvCre M1 preDTX

- 0.100 non --> 0 and 1
- 0.100 per --> 8 to 15
- 0.100 sin --> 16 to 19

## 1yrShamPvCre M1 postDTX
 
- 0.100 non --> 1 and 2
- 0.100 per --> 9 to 16
- 0.100 sin --> 17 to 19

## 1-6 RAP M1 preDTX

- Caveat similar to 1yrShamPvCre-1 where analyzed with previous model.

- 0.100 non --> 0 to 2
    - Hip height noticeably higher at ~ 2.0
    - vid 2 isn't greatest consider excluding
- 0.100 per --> 10 to 17
    - vid 10 may be okay maybe not
    - on vid 11 there might be a artifact from dlc or a fall, unclear.
    - vid 13 for per is pretty solid for a front runner
    - vid 17 also seems solid
- 0.100 sin --> 18 to 20
    - Don't seem too bad
