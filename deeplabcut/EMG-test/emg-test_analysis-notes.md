

# Notes regarding EMG-test Analysis

At this time I am using videos from:
    - M1-pre-emg: vids 3, 4
    - M2-pre-emg: vids 0, 1, 2
    - M1-post-emg: vids 0, 1, 2, 3
    - M2-post-emg: vids 1, 8, 9

## For manuscript

Need to finish this paragraph.

Implantation of EMG electrodes in the mice and cats and wearing reflexive markers by cats could affect walking characteristics of animals. For example, cats in initial stages of training to wear the markers crouch during walking. To address this issue, we compared the step width and MoS before and after implantation of EMG electrodes in two additional mice walking on treadmill at 0.1 m/s and in 2 cats from this study walking overground with a self-selected speed (#.## - #.## m/s). The hindlimb step width decreased post implantation in cat 1 (5.1±0.9 cm vs 4.0±0.9 cm (22%), n1=29, n2=27, p < 0.05, t-test) and cat 2 (3.7±0.9 cm vs. 3.3±1.1 cm (11%), n1=51, n2=60, p < 0.05, t-test). The left MoS decreased in cat 1 (0.86±0.31 cm vs. 0.56±0.37 cm (35%), n1=29, n2=27, p < 0.05, t-test) and did not change in cat 2 (0.74±0.45 cm vs. 0.68±0.47 cm, n1=51, n2=60, p=0.610, t-test). In mice, a widening of the hindlimb step width was observed in mouse 2 post-implantation (2.1 $\pm$ 0.16 cm vs 2.4 $\pm$ 0.22 cm, n1=40, n2=42, p < 0.05, t-test), whereas mouse 1 saw no significant change (2.24 $\pm$ 0.26 cm vs 2.32 $\pm$ 0.31 cm, n1=36, n2=46, p = 0.19, t-test). However, in MoS both mice showed a significant increase in their right MoS (0.86 $\pm$ 0.16 cm vs 1.08 $\pm$ 0.8 cm, n1=27, n2=32, p < 0.05, t-test) for moues 1 and (0.80 $\pm$ 0.16 cm vs 1.06 $\pm$ 0.9 cm, n1=18, n240, p < 0.05, t-test) for mouse 2.

- Hindlimb step width
- MoS changes?
- What t-tests are used?
    - Assuming the n values given are for each of the animals

Key things

- M2's hindlimb step width really all that changes for step width
- Both M1 and M2 vary in their R MoS values.


# Stats

For reference from plots

```
p-value annotation legend:
      ns: 5.00e-02 < p <= 1.00e+00
       *: 1.00e-02 < p <= 5.00e-02
      **: 1.00e-03 < p <= 1.00e-02
     ***: 1.00e-04 < p <= 1.00e-03
    ****: p <= 1.00e-04
``````

### Step Width

Hindlimb_M1 Pre-EMG vs. Hindlimb_M1 Post-EMG: Welch's t-test independent samples, P_val:1.915e-01 t=-1.317e+00

Forelimb_M1 Pre-EMG vs. Forelimb_M1 Post-EMG: Welch's t-test independent samples, P_val:4.270e-01 t=-7.984e-01

Hindlimb_M2 Pre-EMG vs. Hindlimb_M2 Post-EMG: Welch's t-test independent samples, P_val:4.766e-09 t=-6.639e+00

Forelimb_M2 Pre-EMG vs. Forelimb_M2 Post-EMG: Welch's t-test independent samples, P_val:6.955e-01 t=-3.930e-01

Descriptives for M1 Pre-EMG
Forelimb mean: 1.209671440075363 std: 0.22555579892835514 n: 35
Descriptives for M1 Post-EMG
Forelimb mean: 1.2560983889847792 std: 0.294539555786696 n: 46


Descriptives for M1 Pre-EMG
Hindlimb mean: 2.236778222131299 sd: 0.25837368292886714 n: 36
Descriptives for M1 Post-EMG
Hindlimb mean: 2.32140608875234 sd: 0.31195817394979714 n: 46


Descriptives for M2 Pre-EMG
Forelimb mean: 1.1745578789413973 std: 0.3381040566122449 n: 41
Descriptives for M2 Post-EMG
Forelimb mean: 1.200988306759846 std: 0.25542820309387454 n: 41


Descriptives for M2 Pre-EMG
Hindlimb mean: 2.095599033155942 sd: 0.15781075098958902 n: 40
Descriptives for M2 Post-EMG
Hindlimb mean: 2.376247001360333 sd: 0.21581201009452963 n: 42

### MoS

Right_M1 Pre-EMG vs. Right_M1 Post-EMG: Welch's t-test independent samples, P_val:5.194e-08 t=-6.756e+00

Left_M1 Pre-EMG vs. Left_M1 Post-EMG: Welch's t-test independent samples, P_val:3.443e-01 t=-9.532e-01

Right_M2 Pre-EMG vs. Right_M2 Post-EMG: Welch's t-test independent samples, P_val:3.666e-05 t=-5.261e+00

Left_M2 Pre-EMG vs. Left_M2 Post-EMG: Welch's t-test independent samples, P_val:4.721e-01 t=7.320e-01


Descriptives for M1 Pre-EMG Left MoS
mean: 0.6295478234129572 std: 0.11809906561295615 n: 29
Descriptives for M2 Pre-EMG Left MoS
mean: 0.6133980821766996 std: 0.18336058207608197 n: 17

Descriptives for M1 Post-EMG Left MoS
mean: 0.6578265317252469 std: 0.11726816047114733 n: 37
Descriptives for M2 Post-EMG Left MoS
mean: 0.5771773028984456 std: 0.11024329472674756 n: 36

Descriptives for M1 Pre-EMG Right MoS
mean: 0.8565308322469695 std: 0.1509032173407959 n: 27
Descriptives for M1 Post-EMG Right MoS
mean: 1.0791309335113808 std: 0.0806423421942663 n: 32

[Descriptives](Descriptives.md) for M2 Pre-EMG Right MoS
mean: 0.8060872155035476 std: 0.18908321341067544 n: 18
Descriptives for M2 Post-EMG Right MoS
mean: 1.0582171120180237 std: 0.08681537472200468 n: 40


