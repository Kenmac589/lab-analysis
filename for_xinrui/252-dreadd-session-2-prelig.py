import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import sig as sig


def truncate(n, decimals=0):
    multiplier = 10**decimals
    return int(n * multiplier) / multiplier


def show_stims(stim_events, col_max):

    n_plots = len(stim_events)
    col_max = 9
    n_rows = (n_plots + 1) // col_max

    fig, axs = plt.subplots(col_max, n_rows)


def get_stim_region(input_df, stim_event, emg_channel, sampling_rate=1000):

    event_marker = 1
    input_df_subset = input_df.loc[
        :,
        [
            "Time",
            stim_event,
            emg_channel,
        ],
    ]
    input_df_subset = input_df_subset.set_index("Time")
    # print(input_df_subset)
    # stim_start_timinings = input_df_subset.loc[
    #     input_df_subset[stim_event] == event_marker
    # ]

    stim_start_timinings = input_df_subset.loc[
        input_df_subset[stim_event] == event_marker
    ].index.tolist()

    emg_ch = input_df_subset[emg_channel]

    n_plots = len(stim_start_timinings)
    col_max = 9
    n_rows = (n_plots + 1) // col_max

    fig, axs = plt.subplots(col_max, n_rows)

    for i in range(len(stim_start_timinings)):
        # Section up Stimulation region (0.1 s)
        stim_event_end = stim_start_timinings[i] + 0.1
        stim_event_begin = stim_event_end - 0.1
        stim_event_end = truncate(stim_event_end, 3)

        # Section up region looking for cross reflex (0.5 s)
        reflex_region = stim_event_end + 0.5
        reflex_region = truncate(reflex_region, 3)

        stim_region = emg_ch[stim_event_begin:stim_event_end].to_numpy(dtype=float)
        region_oi = emg_ch[stim_event_end:reflex_region].to_numpy(dtype=float)

        # Get background noise region
        peaks, properties = sig.find_peaks(region_oi, prominence=(None, 0.0))

        plt.plot(region_oi)
        plt.plot(peaks, region_oi[peaks], "x")
        plt.show()
        # print(input_df_subset.loc[input_df_subset[stim_event_end] == reflex_region])

    stim_regions = stim_start_timinings

    return stim_regions


def stim_test(input_df, stim_event, emg_channel, sampling_rate=1000):

    event_marker = 1
    input_df_subset = input_df.loc[
        :,
        [
            "Time",
            stim_event,
            emg_channel,
        ],
    ]
    input_df_subset = input_df_subset.set_index("Time")
    # print(input_df_subset)
    # stim_start_timinings = input_df_subset.loc[
    #     input_df_subset[stim_event] == event_marker
    # ]

    stim_start_timinings = input_df_subset.loc[
        input_df_subset[stim_event] == event_marker
    ].index.tolist()

    emg_ch = input_df_subset[emg_channel]

    # Section up Stimulation region (0.1 s)
    stim_event = stim_start_timinings[4]
    region_after = stim_start_timinings[4] + 0.05
    region_before = stim_start_timinings[4] - 0.05
    # stim_event_begin = stim_region_after - 0.01
    region_before = truncate(region_before, 3)
    region_after = truncate(region_after, 3)

    # Section up region looking for cross reflex (0.5 s)
    # reflex_region = region_after + 0.05
    # reflex_region = truncate(reflex_region, 3)

    total_region = emg_ch[region_before:region_after].to_numpy(dtype=float)
    stim_region_before = emg_ch[region_before:stim_event].to_numpy(dtype=float)
    stim_region_after = emg_ch[stim_event:region_after].to_numpy(dtype=float)
    # region_oi = emg_ch[stim_region_after:reflex_region].to_numpy(dtype=float)

    # Setting up plot

    # Get background noise region before and after
    peaks_before, properties_before = sig.find_peaks(
        stim_region_before, prominence=(None, 0.001)
    )
    background_regions_before = stim_region_before[peaks_before]
    background_threshold_before = np.mean(background_regions_before)

    peaks_after, properties_after = sig.find_peaks(
        stim_region_after, prominence=(None, 0.001)
    )
    background_regions_after = stim_region_after[peaks_after]
    background_threshold_after = np.mean(background_regions_after)

    fig = plt.figure(figsize=(7, 7), layout="constrained")
    axs = fig.subplot_mosaic([["total", "total"], ["region_before", "region_after"]])

    axs["total"].plot(total_region)
    axs["total"].axvline(len(total_region) // 2, color="r")

    axs["region_before"].plot(stim_region_before)
    axs["region_before"].plot(peaks_before, stim_region_before[peaks_before], "x")
    axs["region_before"].axhline(background_threshold_before, color="r", linestyle="-")

    axs["region_after"].plot(stim_region_after)
    axs["region_after"].plot(peaks_after, stim_region_after[peaks_after], "x")
    axs["region_after"].axhline(background_threshold_after, color="r", linestyle="-")
    plt.show()
    # print(input_df_subset.loc[input_df_subset[stim_event_end] == reflex_region])

    stim_regions = stim_start_timinings

    return stim_regions


spike_df = pd.read_csv("./252-dreadd-session-2-pre-ligand.txt", delimiter=",", header=0)


# Nerve Event Channel
nerve_event = "17 L stim"
emg_channel = "10 Right TA"

time = spike_df["Time"]
nerve_stim = spike_df["17 L stim"]
ta_emg = spike_df["10 Right TA"]


func_test = stim_test(spike_df, nerve_event, emg_channel)

# time = func_test["Time"]
# nerve_stim = func_test["17 L stim"]
# ta_emg = func_test["10 Right TA"]
#
#
# axs[0].plot(nerve_stim, "g-", label="Nerve Stimulation")
# axs[1].plot(ta_emg, "b-", label="Right TA")
# fig.suptitle("Right TA Activity with Nerve Stimulation")
# plt.show()
