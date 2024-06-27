import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from pandas import DataFrame as df
from statannotations.Annotator import Annotator

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


def step_width_batch_est(inputdf, x_channels, y_channels):
    """Doing step width calculation in one go
    :param inputdf: A spike file input as *.csv or formatted as such
    :param event_channels: A list with all the proper channel names for the event channels
    :param y_channels: A list with all the proper channel names for the channels from DLC
    :note: The proper order for event channels goes lhl, lfl, rhl, rfl with swonset first.

    :return fl_step_widths: array of step width values for the forelimb
    :return hl_step_widths: array of step width values for the hindlimb
    """

    # For forelimb
    fl_step_widths = ls.step_width_est(
        inputdf,
        rl_x=x_channels[0],
        ll_x=x_channels[1],
        rl_y=y_channels[0],
        ll_y=y_channels[1],
    )
    hl_step_widths = ls.step_width_est(
        inputdf,
        rl_x=x_channels[2],
        ll_x=x_channels[3],
        rl_y=y_channels[2],
        ll_y=y_channels[3],
    )

    return fl_step_widths, hl_step_widths


def sw_condition_add(
    input_df, file_list, x_chan, y_chan, condition, limb, perturbation_state
):
    for i in range(len(file_list)):
        current_file = pd.read_csv(file_list[i])
        step_width_values = ls.step_width_est(
            current_file,
            rl_x=x_chan[0],
            ll_x=x_chan[1],
            rl_y=y_chan[0],
            ll_y=y_chan[1],
        )
        limb = limb
        perturbation_state = perturbation_state
        step_width_values = np.array(step_width_values, dtype=float)
        # print(len(step_width_values))
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

conditions = [
    "Non-Perturbation",
    "Perturbation",
    "Sinusoidal",
]

wt_y_channels = ["35 FRy (cm)", "33 FLy (cm)", "30 HRy (cm)", "28 HLy (cm)"]
wt_x_channels = ["34 FRx (cm)", "32 FLx (cm)", "29 HRx (cm)", "27 HLx (cm)"]

wt_fl_x_channels = ["34 FRx (cm)", "32 FLx (cm)"]
wt_fl_y_channels = ["35 FRy (cm)", "33 FLy (cm)"]
wt_hl_x_channels = ["29 HRx (cm)", "27 HLx (cm)"]
wt_hl_y_channels = ["30 HRy (cm)", "28 HLy (cm)"]

dtr_fl_x_channels = ["39 FRx (cm)", "37 FLx (cm)"]
dtr_fl_y_channels = ["40 FRy (cm)", "38 FLy (cm)"]
dtr_hl_x_channels = ["35 HRx (cm)", "33 HLx (cm)"]
dtr_hl_y_channels = ["36 HRy (cm)", "34 HLy (cm)"]

wt_non = [
    "./wt_data/wt-1-non-all.txt",
    "./wt_data/wt-2-non-all.txt",
    "./wt_data/wt-3-non-all.txt",
    "./wt_data/wt-4-non-all.txt",
    "./wt_data/wt-5-non-all.txt",
]
wt_per = [
    "./wt_data/wt-1-per-all.txt",
    "./wt_data/wt-2-per-all.txt",
    "./wt_data/wt-3-per-all.txt",
    "./wt_data/wt-4-per-all.txt",
    "./wt_data/wt-5-per-all.txt",
]
wt_sin = [
    "./wt_data/wt-1-sin-all.txt",
    "./wt_data/wt-2-sin-all.txt",
    "./wt_data/wt-3-sin-all.txt",
    "./wt_data/wt-4-sin-all.txt",
]

egr3_non = [
    "./egr3_data/egr3-6-non-all.txt",
    "./egr3_data/egr3-7-non-all.txt",
    "./egr3_data/egr3-8-non-all.txt",
    "./egr3_data/egr3-9-non-all.txt",
]
egr3_per = [
    "./egr3_data/egr3-6-per-all.txt",
    "./egr3_data/egr3-7-per-all.txt",
    "./egr3_data/egr3-8-per-all.txt",
    "./egr3_data/egr3-9-per-all-2.txt",
]
egr3_sin = [
    "./egr3_data/egr3-6-sin-all.txt",
    "./egr3_data/egr3-7-sin-all.txt",
    "./egr3_data/egr3-8-sin-all.txt",
    "./egr3_data/egr3-9-sin-all-1.txt",
]

dtrpre_non = [
    "./dtr_data/predtx/dtr-pre-2-non-all.txt",
    "./dtr_data/predtx/dtr-pre-3-non-all.txt",
    "./dtr_data/predtx/dtr-pre-5-non-all.txt",
    "./dtr_data/predtx/dtr-pre-6-non-all.txt",
    "./dtr_data/predtx/dtr-pre-7-non-all.txt",
]
dtrpre_per = [
    "./dtr_data/predtx/dtr-pre-2-per-all.txt",
    "./dtr_data/predtx/dtr-pre-3-per-all.txt",
    "./dtr_data/predtx/dtr-pre-5-per-all-1.txt",
    "./dtr_data/predtx/dtr-pre-5-per-all-2.txt",
    "./dtr_data/predtx/dtr-pre-6-per-all.txt",
    "./dtr_data/predtx/dtr-pre-7-per-all.txt",
]
dtrpre_sin = [
    "./dtr_data/predtx/dtr-pre-2-sin-all.txt",
    "./dtr_data/predtx/dtr-pre-3-sin-all.txt",
    "./dtr_data/predtx/dtr-pre-5-sin-all.txt",
    "./dtr_data/predtx/dtr-pre-6-sin-all.txt",
    "./dtr_data/predtx/dtr-pre-7-sin-all.txt",
]

dtrpost_non = [
    "./dtr_data/postdtx/dtr-post-2-non-all.txt",
    "./dtr_data/postdtx/dtr-post-3-non-all.txt",
    "./dtr_data/postdtx/dtr-post-5-non-all.txt",
    "./dtr_data/postdtx/dtr-post-6-non-all.txt",
    "./dtr_data/postdtx/dtr-post-8-non-all.txt",
]
dtrpost_per = [
    "./dtr_data/postdtx/dtr-post-2-per-all.txt",
    "./dtr_data/postdtx/dtr-post-3-per-all.txt",
    "./dtr_data/postdtx/dtr-post-5-per-all.txt",
    "./dtr_data/postdtx/dtr-post-6-per-all.txt",
]
dtrpost_sin = [
    "./dtr_data/postdtx/dtr-post-2-sin-all.txt",
    "./dtr_data/postdtx/dtr-post-3-sin-all.txt",
    "./dtr_data/postdtx/dtr-post-5-sin-all.txt",
    "./dtr_data/postdtx/dtr-post-6-sin-all.txt",
]

step_width_df = sw_condition_add(
    step_width_df,
    wt_non,
    wt_fl_x_channels,
    wt_fl_y_channels,
    "WT",
    "Forelimb",
    "Non-Perturbation",
)
step_width_df = sw_condition_add(
    step_width_df,
    wt_non,
    wt_hl_x_channels,
    wt_hl_y_channels,
    "WT",
    "Hindlimb",
    "Non-Perturbation",
)
step_width_df = sw_condition_add(
    step_width_df,
    wt_per,
    wt_fl_x_channels,
    wt_fl_y_channels,
    "WT",
    "Forelimb",
    "Perturbation",
)
step_width_df = sw_condition_add(
    step_width_df,
    wt_per,
    wt_hl_x_channels,
    wt_hl_y_channels,
    "WT",
    "Hindlimb",
    "Perturbation",
)
step_width_df = sw_condition_add(
    step_width_df,
    wt_sin,
    wt_fl_x_channels,
    wt_fl_y_channels,
    "WT",
    "Forelimb",
    "Sinusoidal",
)
step_width_df = sw_condition_add(
    step_width_df,
    wt_sin,
    wt_hl_x_channels,
    wt_hl_y_channels,
    "WT",
    "Hindlimb",
    "Sinusoidal",
)

step_width_df = sw_condition_add(
    step_width_df,
    egr3_non,
    wt_fl_x_channels,
    wt_fl_y_channels,
    "Egr3",
    "Forelimb",
    "Non-Perturbation",
)
step_width_df = sw_condition_add(
    step_width_df,
    egr3_non,
    wt_hl_x_channels,
    wt_hl_y_channels,
    "Egr3",
    "Hindlimb",
    "Non-Perturbation",
)
step_width_df = sw_condition_add(
    step_width_df,
    egr3_per,
    wt_fl_x_channels,
    wt_fl_y_channels,
    "Egr3",
    "Forelimb",
    "Perturbation",
)
step_width_df = sw_condition_add(
    step_width_df,
    egr3_per,
    wt_hl_x_channels,
    wt_hl_y_channels,
    "Egr3",
    "Hindlimb",
    "Perturbation",
)
step_width_df = sw_condition_add(
    step_width_df,
    egr3_sin,
    wt_fl_x_channels,
    wt_fl_y_channels,
    "Egr3",
    "Forelimb",
    "Sinusoidal",
)
step_width_df = sw_condition_add(
    step_width_df,
    egr3_sin,
    wt_hl_x_channels,
    wt_hl_y_channels,
    "Egr3",
    "Hindlimb",
    "Sinusoidal",
)

step_width_df = sw_condition_add(
    step_width_df,
    dtrpre_non,
    dtr_fl_x_channels,
    dtr_fl_y_channels,
    "Pre-DTX",
    "Forelimb",
    "Non-Perturbation",
)
step_width_df = sw_condition_add(
    step_width_df,
    dtrpre_non,
    dtr_hl_x_channels,
    dtr_hl_y_channels,
    "Pre-DTX",
    "Hindlimb",
    "Non-Perturbation",
)
step_width_df = sw_condition_add(
    step_width_df,
    dtrpre_per,
    dtr_fl_x_channels,
    dtr_fl_y_channels,
    "Pre-DTX",
    "Forelimb",
    "Perturbation",
)
step_width_df = sw_condition_add(
    step_width_df,
    dtrpre_per,
    dtr_hl_x_channels,
    dtr_hl_y_channels,
    "Pre-DTX",
    "Hindlimb",
    "Perturbation",
)
step_width_df = sw_condition_add(
    step_width_df,
    dtrpre_sin,
    dtr_fl_x_channels,
    dtr_fl_y_channels,
    "Pre-DTX",
    "Forelimb",
    "Sinusoidal",
)
step_width_df = sw_condition_add(
    step_width_df,
    dtrpre_sin,
    dtr_hl_x_channels,
    dtr_hl_y_channels,
    "Pre-DTX",
    "Hindlimb",
    "Sinusoidal",
)

step_width_df = sw_condition_add(
    step_width_df,
    dtrpost_non,
    dtr_fl_x_channels,
    dtr_fl_y_channels,
    "Post-DTX",
    "Forelimb",
    "Non-Perturbation",
)
step_width_df = sw_condition_add(
    step_width_df,
    dtrpost_non,
    dtr_hl_x_channels,
    dtr_hl_y_channels,
    "Post-DTX",
    "Hindlimb",
    "Non-Perturbation",
)
step_width_df = sw_condition_add(
    step_width_df,
    dtrpost_per,
    dtr_fl_x_channels,
    dtr_fl_y_channels,
    "Post-DTX",
    "Forelimb",
    "Perturbation",
)
step_width_df = sw_condition_add(
    step_width_df,
    dtrpost_per,
    dtr_hl_x_channels,
    dtr_hl_y_channels,
    "Post-DTX",
    "Hindlimb",
    "Perturbation",
)
step_width_df = sw_condition_add(
    step_width_df,
    dtrpost_sin,
    dtr_fl_x_channels,
    dtr_fl_y_channels,
    "Post-DTX",
    "Forelimb",
    "Sinusoidal",
)
step_width_df = sw_condition_add(
    step_width_df,
    dtrpost_sin,
    dtr_hl_x_channels,
    dtr_hl_y_channels,
    "Post-DTX",
    "Hindlimb",
    "Sinusoidal",
)

sw_combo = step_width_df.drop(columns=["Limb"])
con_sw_combo = step_width_df.drop(columns=["Limb"])

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set(style="white", font="serif", font_scale=1.7, palette="colorblind", rc=custom_params)

combo_pairs = [
    [("Non-Perturbation"), ("Perturbation")],
]
combo_legend = ["Non-Perturbation", "Perturbation", "Sinusoidal"]

perturbation_state_order = ["Non-Perturbation", "Perturbation", "Sinusoidal"]

# plt.title("Step Width between WT, Egr3 KO, and DTX Mice Pre and Post Injection")
condition_pairs = [
    # Comparison within wildtype condition
    [("WT", "Non-Perturbation"), ("WT", "Perturbation")],
    [("WT", "Sinusoidal"), ("WT", "Perturbation")],
    [("WT", "Non-Perturbation"), ("WT", "Sinusoidal")],
    # Comparison between Wildtype and Pre-DTX
    [("WT", "Non-Perturbation"), ("Pre-DTX", "Non-Perturbation")],
    [("WT", "Sinusoidal"), ("Pre-DTX", "Sinusoidal")],
    [("WT", "Perturbation"), ("Pre-DTX", "Perturbation")],
    # Comparison within Egr3 condition
    [("Egr3", "Non-Perturbation"), ("Egr3", "Perturbation")],
    [("Egr3", "Sinusoidal"), ("Egr3", "Perturbation")],
    [("Egr3", "Non-Perturbation"), ("Egr3", "Sinusoidal")],
    # Comparison within Pre-DTX condition
    [("Pre-DTX", "Non-Perturbation"), ("Pre-DTX", "Perturbation")],
    [("Pre-DTX", "Sinusoidal"), ("Pre-DTX", "Perturbation")],
    [("Pre-DTX", "Non-Perturbation"), ("Pre-DTX", "Sinusoidal")],
    # Comparison within Post-DTX condition
    [("Post-DTX", "Non-Perturbation"), ("Post-DTX", "Perturbation")],
    [("Post-DTX", "Sinusoidal"), ("Post-DTX", "Perturbation")],
    [("Post-DTX", "Non-Perturbation"), ("Post-DTX", "Sinusoidal")],
]

perturbation_state_order = ["Non-Perturbation", "Perturbation", "Sinusoidal"]
cond_combo_plot_params = {
    "data": con_sw_combo,
    "x": "Condition",
    "y": "Step Width",
    "hue": "Perturbation State",
    "hue_order": perturbation_state_order,
    # "inner": "point",
}

# axs[0].set_title("MoS between conditions")
cond_combo_comp = sns.barplot(**cond_combo_plot_params, ci=95, capsize=0.05)
plt.legend(loc="upper right", fontsize=16)
annotator = Annotator(cond_combo_comp, condition_pairs, **cond_combo_plot_params)
annotator.new_plot(
    cond_combo_comp, condition_pairs, plot="barplot", **cond_combo_plot_params
)
annotator.configure(
    hide_non_significant=True, test="t-test_ind", text_format="star", loc="inside"
)

annotator.apply_test().annotate(line_offset_to_group=0.2, line_offset=0.1)

fig = mpl.pyplot.gcf()
fig.set_size_inches(19.8, 10.80)
fig.tight_layout()
plt.savefig("./combined_figures/step_widths_all.png", dpi=300)
# plt.show()
