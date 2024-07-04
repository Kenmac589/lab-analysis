import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import latstability as ls


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
    print(input_df_subset.head(5))
    input_df_subset = input_df_subset.set_index("Time")
    # print(input_df_subset)
    # stim_start_timinings = input_df_subset.loc[
    #     input_df_subset[stim_event] == event_marker
    # ]

    stim_start_timinings = input_df_subset.loc[
        input_df_subset[stim_event] == event_marker
    ].index.tolist()

    stim_regions = stim_start_timinings

    return stim_regions


spike_df = pd.read_csv("./252-dreadd-session-2-pre-ligand.txt", delimiter=",", header=0)


fig, axs = plt.subplots(2)

# Nerve Event Channel
nerve_event = "17 L stim"
emg_channel = "10 Right TA"

time = spike_df["Time"]
nerve_stim = spike_df["17 L stim"]
ta_emg = spike_df["10 Right TA"]

func_test = get_stim_region(spike_df, nerve_event, emg_channel)
print(func_test)

# time = func_test["Time"]
# nerve_stim = func_test["17 L stim"]
# ta_emg = func_test["10 Right TA"]
#
#
# axs[0].plot(nerve_stim, "g-", label="Nerve Stimulation")
# axs[1].plot(ta_emg, "b-", label="Right TA")
# fig.suptitle("Right TA Activity with Nerve Stimulation")
# plt.show()
