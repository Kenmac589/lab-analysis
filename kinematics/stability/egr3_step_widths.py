import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statannotations import Annotator

import latstability as ls


def step_width_batch(inputdf, event_channels, y_channels):
    """Doing step width calculation in one go
    :param inputdf: A spike file input as *.csv or formatted as such
    :param event_channels: A list with all the proper channel names for the event channels
    :param y_channels: A list with all the proper channel names for the channels from DLC
    :note: The proper order for event channels goes lhl, lfl, rhl, rfl with swonset first.

    :return fl_step_widths: array of step width values for the forelimb
    :return hl_step_widths: array of step width values for the hindlimb
    """

    lhl_st_lengths, lhl_st_timings = ls.stance_duration(
        inputdf, swonset_channel=event_channels[0], swoffset_channel=event_channels[1]
    )
    lfl_st_lengths, lfl_st_timings = ls.stance_duration(
        inputdf, swonset_channel=event_channels[2], swoffset_channel=event_channels[3]
    )
    rhl_st_lengths, rhl_st_timings = ls.stance_duration(
        inputdf, swonset_channel=event_channels[4], swoffset_channel=event_channels[5]
    )
    rfl_st_lengths, rfl_st_timings = ls.stance_duration(
        inputdf, swonset_channel=event_channels[6], swoffset_channel=event_channels[7]
    )

    # For forelimb
    fl_step_widths = ls.step_width(
        inputdf, rfl_st_timings, lfl_st_timings, rl_y="35 FRy (cm)", ll_y="33 FLy (cm)"
    )
    hl_step_widths = ls.step_width(
        inputdf, rhl_st_timings, lhl_st_timings, rl_y="30 HRy (cm)", ll_y="28 HLy (cm)"
    )

    return fl_step_widths, hl_step_widths


# Wild type data
wt_event_channels = [
    "51 HLl Sw on",
    "52 HLl Sw of",
    "55 FLl Sw on",
    "56 FLl Sw of",
    "53 HLr Sw on",
    "54 HLr Sw of",
    "57 FLr Sw on",
    "58 FLr Sw of",
]

wt_2_event_channels = [
    "51 lHLswon",
    "52 lHLswoff",
    "55 lFLswon",
    "56 lFLswoff",
    "53 rHLswon",
    "54 rHLswoff",
    "57 rFLswon",
    "58 rFLswoff",
]

wt_3_event_channels = [
    "56 lHL swon",
    "57 lHL swoff",
    "52 lFL swon",
    "53 lFL swoff",
    "58 rHL swon",
    "59 rHL swoff",
    "54 rFL swon",
    "55 rFL swoff",
]

wt_4_event_channels = [
    "57 lHL swon",
    "58 lHL swoff",
    "53 lFL swon",
    "54 lFL swoff",
    "55 rHL swon",
    "56 rHL swoff",
    "51 rFL swon",
    "52 rFL swoff",
]

wt_5_event_channels = [
    "48 lHL swon",
    "49 lHL swoff",
    "44 lFL swon",
    "45 lFL swoff",
    "46 rHL swon",
    "47 rHL swoff",
    "42 rFL swon",
    "43 rFL swoff",
]


wt_y_channels = ["35 FRy (cm)", "33 FLy (cm)", "30 HRy (cm)", "28 HLy (cm)"]

wt1nondf = pd.read_csv("./wt_data/wt-1-non-all.txt", delimiter=",", header=0)
wt1perdf = pd.read_csv("./wt_data/wt-1-per-all.txt", delimiter=",", header=0)
wt1sindf = pd.read_csv("./wt_data/wt-1-sin-all.txt", delimiter=",", header=0)
wt2nondf = pd.read_csv("./wt_data/wt-2-non-all.txt", delimiter=",", header=0)
wt2perdf = pd.read_csv("./wt_data/wt-2-per-all.txt", delimiter=",", header=0)
wt2sindf = pd.read_csv("./wt_data/wt-2-sin-all.txt", delimiter=",", header=0)
wt3nondf = pd.read_csv("./wt_data/wt-3-non-all.txt", delimiter=",", header=0)
wt3perdf = pd.read_csv("./wt_data/wt-3-per-all.txt", delimiter=",", header=0)
wt3sindf = pd.read_csv("./wt_data/wt-3-sin-all.txt", delimiter=",", header=0)
wt4nondf = pd.read_csv("./wt_data/wt-4-non-all.txt", delimiter=",", header=0)
wt4perdf = pd.read_csv("./wt_data/wt-4-per-all.txt", delimiter=",", header=0)
wt4sindf = pd.read_csv("./wt_data/wt-4-sin-all.txt", delimiter=",", header=0)
wt5nondf = pd.read_csv("./wt_data/wt-5-non-all.txt", delimiter=",", header=0)
wt5perdf = pd.read_csv("./wt_data/wt-5-per-all.txt", delimiter=",", header=0)

# Step Width Calculation

# Non-Perturbation
wt_1_non_fl, wt_1_non_hl = step_width_batch(wt1nondf, wt_event_channels, wt_y_channels)
wt_2_non_fl, wt_2_non_hl = step_width_batch(
    wt2nondf, wt_2_event_channels, wt_y_channels
)
wt_3_non_fl, wt_3_non_hl = step_width_batch(
    wt3nondf, wt_3_event_channels, wt_y_channels
)
wt_4_non_fl, wt_4_non_hl = step_width_batch(
    wt4nondf, wt_4_event_channels, wt_y_channels
)
wt_5_non_fl, wt_5_non_hl = step_width_batch(
    wt5nondf, wt_5_event_channels, wt_y_channels
)

# NOTE: If you are looking at this later, it is alway fl then hl in presentation
print("Non-Perturbation")
print()
print("M1")
print(wt_1_non_fl)
print(wt_1_non_hl)
print()
print("M2")
print(wt_2_non_fl)
print(wt_2_non_hl)
print()
print("M3")
print(wt_3_non_fl)
print(wt_3_non_hl)
print()
print("M4")
print(wt_4_non_fl)
print(wt_4_non_hl)
print()
print("M5")
print(wt_5_non_fl)
print(wt_5_non_hl)
print()


# Perturbation
wt_1_per_fl, wt_1_per_hl = step_width_batch(wt1perdf, wt_event_channels, wt_y_channels)
wt_2_per_fl, wt_2_per_hl = step_width_batch(
    wt2perdf, wt_2_event_channels, wt_y_channels
)
wt_3_per_fl, wt_3_per_hl = step_width_batch(
    wt3perdf, wt_3_event_channels, wt_y_channels
)
wt_4_per_fl, wt_4_per_hl = step_width_batch(
    wt4perdf, wt_4_event_channels, wt_y_channels
)
wt_5_per_fl, wt_5_per_hl = step_width_batch(
    wt5perdf, wt_5_event_channels, wt_y_channels
)

print("Perturbation")
print()
print("M1")
print(wt_1_per_fl)
print(wt_1_per_hl)
print()
print("M2")
print(wt_2_per_fl)
print(wt_2_per_hl)
print()
print("M3")
print(wt_3_per_fl)
print(wt_3_per_hl)
print()
print("M4")
print(wt_4_per_fl)
print(wt_4_per_hl)
print()
print("M5")
print(wt_5_per_fl)
print(wt_5_per_hl)
print()


wt_1_sin_fl, wt_1_sin_hl = step_width_batch(wt1sindf, wt_event_channels, wt_y_channels)
wt_2_sin_fl, wt_2_sin_hl = step_width_batch(
    wt2sindf, wt_2_event_channels, wt_y_channels
)
wt_3_sin_fl, wt_3_sin_hl = step_width_batch(
    wt3sindf, wt_3_event_channels, wt_y_channels
)
wt_4_sin_fl, wt_4_sin_hl = step_width_batch(
    wt4sindf, wt_4_event_channels, wt_y_channels
)

print("Sinusoidal")
print()
print("M1")
print(wt_1_sin_fl)
print(wt_1_sin_hl)
print()
print("M2")
print(wt_2_sin_fl)
print(wt_2_sin_hl)
print()
print("M3")
print(wt_3_sin_fl)
print(wt_3_sin_hl)
print()
print("M4")
print(wt_4_sin_fl)
print(wt_4_sin_hl)

#
