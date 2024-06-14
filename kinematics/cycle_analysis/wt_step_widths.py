import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame as df
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

    # For forelimb
    fl_step_widths = ls.step_width(
        inputdf,
        rl_swoff=event_channels[7],
        ll_swoff=event_channels[3],
        rl_y="35 FRy (cm)",
        ll_y="33 FLy (cm)",
    )
    hl_step_widths = ls.step_width(
        inputdf,
        rl_swoff=event_channels[5],
        ll_swoff=event_channels[1],
        rl_y="30 HRy (cm)",
        ll_y="28 HLy (cm)",
    )

    return fl_step_widths, hl_step_widths


def sw_condition_add(input_df, step_width_values, condition, limb, perturbation_state):
    limb = limb
    perturbation_state = perturbation_state
    step_width_values = np.array(step_width_values, dtype=float)
    # step_width_values = step_width_values.ravel()
    for j in range(len(step_width_values)):
        entry = step_width_values[j]
        step_width_entry = [[condition, limb, perturbation_state, entry]]

        input_df = input_df._append(
            pd.DataFrame(
                step_width_entry,
                columns=["Condition", "Limb", "Perturbation State", "Step Width"],
            ),
            ignore_index=True,
        )

    return input_df


step_width_df = df(columns=["Condition", "Limb", "Perturbation State", "Step Width"])

# Wild type data
wt_event_channels = {
    "wt_1": [
        "51 HLl Sw on",
        "52 HLl Sw of",
        "55 FLl Sw on",
        "56 FLl Sw of",
        "53 HLr Sw on",
        "54 HLr Sw of",
        "57 FLr Sw on",
        "58 FLr Sw of",
    ],
    "wt_2": [
        "51 lHLswon",
        "52 lHLswoff",
        "55 lFLswon",
        "56 lFLswoff",
        "53 rHLswon",
        "54 rHLswoff",
        "57 rFLswon",
        "58 rFLswoff",
    ],
    "wt_3": [
        "56 lHL swon",
        "57 lHL swoff",
        "52 lFL swon",
        "53 lFL swoff",
        "58 rHL swon",
        "59 rHL swoff",
        "54 rFL swon",
        "55 rFL swoff",
    ],
    "wt_4": [
        "57 lHL swon",
        "58 lHL swoff",
        "53 lFL swon",
        "54 lFL swoff",
        "55 rHL swon",
        "56 rHL swoff",
        "51 rFL swon",
        "52 rFL swoff",
    ],
    "wt_5": [
        "48 lHL swon",
        "49 lHL swoff",
        "44 lFL swon",
        "45 lFL swoff",
        "46 rHL swon",
        "47 rHL swoff",
        "42 rFL swon",
        "43 rFL swoff",
    ],
}

wt_raw = {
    "wt_1": [
        "./wt_data/wt-1-non-all.txt",
        "./wt_data/wt-1-per-all.txt",
        "./wt_data/wt-1-sin-all.txt",
    ],
    "wt_2": [
        "./wt_data/wt-2-non-all.txt",
        "./wt_data/wt-2-per-all.txt",
        "./wt_data/wt-2-sin-all.txt",
    ],
    "wt_3": [
        "./wt_data/wt-3-non-all.txt",
        "./wt_data/wt-3-per-all.txt",
        "./wt_data/wt-3-sin-all.txt",
    ],
    "wt_4": [
        "./wt_data/wt-4-non-all.txt",
        "./wt_data/wt-4-per-all.txt",
        "./wt_data/wt-4-sin-all.txt",
    ],
}


wt_y_channels = ["35 FRy (cm)", "33 FLy (cm)", "30 HRy (cm)", "28 HLy (cm)"]

wt5nondf = pd.read_csv("./wt_data/wt-5-non-all.txt", delimiter=",", header=0)
wt5perdf = pd.read_csv("./wt_data/wt-5-per-all.txt", delimiter=",", header=0)


# Step Width Calculation
conditions = [
    "Non-Perturbation",
    "Perturbation",
    "Sinusoidal",
]

for i in wt_raw:
    print(f"mouse being considered is {i}")
    mouse_data = wt_raw[i]
    print(wt_event_channels[i])
    for j in range(len(mouse_data)):
        event_channels = wt_event_channels[i]  # getting channel names
        current_condtion = conditions[j]  # simply to keep track
        data_path = mouse_data[j]  # file from spike for condition
        trial_spike = pd.read_csv(data_path, delimiter=",", header=0)

        fl_stepw, hl_stepw = step_width_batch(
            trial_spike, event_channels=event_channels, y_channels=wt_y_channels
        )
        step_width_df = sw_condition_add(
            step_width_df, fl_stepw, "WT", "Forelimb", current_condtion
        )
        step_width_df = sw_condition_add(
            step_width_df, hl_stepw, "WT", "Hindlimb", current_condtion
        )

        print(f"forelimb step width for {current_condtion} is {fl_stepw}")
        print(f"hindlimb step width for {current_condtion} is {hl_stepw}")

    print()

# M5 Does not have a sinusoidal recording so
# Non-Perturbation
wt_5_non_fl, wt_5_non_hl = step_width_batch(
    wt5nondf, wt_event_channels["wt_5"], wt_y_channels
)
step_width_df = sw_condition_add(
    step_width_df, wt_5_non_fl, "WT", "Forelimb", "Non-Perturbation"
)
step_width_df = sw_condition_add(
    step_width_df, wt_5_non_hl, "WT", "Hindlimb", "Non-Perturbation"
)
# Perturbation
wt_5_per_fl, wt_5_per_hl = step_width_batch(
    wt5perdf, wt_event_channels["wt_5"], wt_y_channels
)
step_width_df = sw_condition_add(
    step_width_df, wt_5_per_fl, "WT", "Forelimb", "Perturbation"
)
step_width_df = sw_condition_add(
    step_width_df, wt_5_per_hl, "WT", "Hindlimb", "Perturbation"
)

print(step_width_df)
step_width_df.to_csv("./wt_step_widths.csv")

# # Egr3 Mice
# egr3_event_channels = {
#     "egr3_6": [
#         "51 HLl Sw on",
#         "52 HLl Sw of",
#         "55 FLl Sw on",
#         "56 FLl Sw of",
#         "53 HLr Sw on",
#         "54 HLr Sw of",
#         "57 FLr Sw on",
#         "58 FLr Sw of",
#     ],
#     "egr3_7": [
#         "51 lHLswon",
#         "52 lHLswoff",
#         "55 lFLswon",
#         "56 lFLswoff",
#         "53 rHLswon",
#         "54 rHLswoff",
#         "57 rFLswon",
#         "58 rFLswoff",
#     ],
#     "egr3_8": [
#         "56 lHL swon",
#         "57 lHL swoff",
#         "52 lFL swon",
#         "53 lFL swoff",
#         "58 rHL swon",
#         "59 rHL swoff",
#         "54 rFL swon",
#         "55 rFL swoff",
#     ],
#     "egr3_9": [
#         "57 lHL swon",
#         "58 lHL swoff",
#         "53 lFL swon",
#         "54 lFL swoff",
#         "55 rHL swon",
#         "56 rHL swoff",
#         "51 rFL swon",
#         "52 rFL swoff",
#     ],
#     "egr3_10": [
#         "48 lHL swon",
#         "49 lHL swoff",
#         "44 lFL swon",
#         "45 lFL swoff",
#         "46 rHL swon",
#         "47 rHL swoff",
#         "42 rFL swon",
#         "43 rFL swoff",
#     ],
# }
#
# egr3_raw = {
#     "egr3_6": [
#         "./egr3_data/egr3-6-non-all.txt",
#         "./egr3_data/egr3-6-per-all.txt",
#         "./egr3_data/egr3-6-sin-all.txt",
#     ],
#     "egr3_7": [
#         "./egr3_data/egr3-7-non-all.txt",
#         "./egr3_data/egr3-7-per-all.txt",
#         "./egr3_data/egr3-7-sin-all.txt",
#     ],
#     "egr3_8": [
#         "./egr3_data/egr3-8-non-all.txt",
#         "./egr3_data/egr3-8-per-all.txt",
#         "./egr3_data/egr3-8-sin-all.txt",
#     ],
#     "egr3_9": [
#         "./egr3_data/egr3-9-non-all.txt",
#         "./egr3_data/egr3-9-per-all-2.txt",
#         "./egr3_data/egr3-9-sin-all-1.txt",
#     ],
# }
#
#
# egr3_y_channels = ["35 FRy (cm)", "33 FLy (cm)", "30 HRy (cm)", "28 HLy (cm)"]
#
# for i in egr3_raw:
#     print(f"mouse being considered is {i}")
#     mouse_data = egr3_raw[i]
#     # print(wt_event_channels[i])
#     for j in range(len(mouse_data)):
#         event_channels = egr3_event_channels[i]  # getting channel names
#         current_condtion = conditions[j]  # simply to keep track
#         data_path = mouse_data[j]  # file from spike for condition
#         trial_spike = pd.read_csv(data_path, delimiter=",", header=0)
#         fl_stepw, hl_stepw = step_width_batch(
#             trial_spike, event_channels=event_channels, y_channels=wt_y_channels
#         )
#         print(f"forelimb step width for {current_condtion} is {fl_stepw}")
#         print(f"hindlimb step width for {current_condtion} is {hl_stepw}")
#
#     print()
