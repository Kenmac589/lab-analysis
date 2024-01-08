# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from pandas import DataFrame as df
import seaborn as sns
# from scipy import stats as st
from statannotations.Annotator import Annotator
import motorpyrimitives as mp


# Plotting Section
def sel_primitive_trace(motor_primitives, synergy_selection, selected_primitive_title="Output"):
    """This will plot the selected motor primitives
    @param data_input: path to csv data file
    @param synergy_selection: how many synergies you want

    @return null
    """

    motor_p_data = pd.read_csv(motor_primitives, header=0)

    motor_primitives = motor_p_data.to_numpy()

    samples = np.arange(0, len(motor_primitives))
    samples_binned = np.arange(200)
    number_cycles = len(motor_primitives) // 200

    # Plot
    primitive_trace = np.zeros(200)

    # Plotting Primitive Selected Synergy Count

    # Iterate over the bins
    for i in range(number_cycles):
        # Get the data for the current bin

        time_point_average = motor_primitives[i * 200: (i + 1) * 200, synergy_selection - 1]

        # Accumulate the trace values
        current_primitive = motor_primitives[i * 200: (i + 1) * 200, synergy_selection - 1]

        primitive_mask = current_primitive > 0.0
        # applying mask to exclude values which were subject to rounding errors
        mcurrent_primitive = np.asarray(current_primitive[primitive_mask])

        primitive_trace += time_point_average

    # Calculate the average by dividing the accumulated values by the number of bins
    primitive_trace /= number_cycles

    plt.plot(samples[samples_binned], primitive_trace)

    # Plotting individual primitives in the background
    selected_primitive = motor_primitives[:, synergy_selection - 2]

    # Using the order F so the values are in column order
    binned_primitives_raw = selected_primitive.reshape((200, -1), order='F')
    # binned_primitives = ndimage.median_filter(binned_primitives_raw, size=3)
    # plt.plot(binned_primitives_raw[:, i], color='black', alpha=0.2)
    # plt.plot(binned_primitives_raw, color='black', alpha=0.2)

    # Removing axis values
    plt.xticks([])
    plt.yticks([])

    # Add a vertical line at the halfway point
    plt.axvline(x=100, color='black')

    # Removing top and right spines of the plot
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    plt.title(selected_primitive_title, fontsize=16, fontweight='bold')
    # plt.savefig(selected_primitive_title, dpi=300)
    # plt.show()

    # fwhl = np.asarray(fwhl)
    # return fwhllab

# %%
# ================
# Synergy analysis
# ================

synergy_selection = 3

primtitives_wt_non = [
    './wt-m1-non-primitives.txt',
    './wt-m2-non-primitives.txt',
    './wt-m3-non-primitives.txt',
    './wt-m4-non-primitives.txt',
    './wt-m5-non-primitives.txt',

]

primtitives_wt_per = [
    './wt-m1-per-primitives.txt',
    './wt-m2-per-primitives.txt',
    './wt-m3-per-primitives.txt',
    './wt-m4-per-primitives.txt',
    './wt-m5-per-primitives.txt',
]

modules_wt = [
    './wt_data/wt_m1_non_modules.csv',
    './wt_data/wt_m1_per_modules.csv',
    './wt_data/wt_m2_non_modules.csv',
    './wt_data/wt_m2_per_modules.csv',
    './wt_data/wt_m3_non_modules.csv',
    './wt_data/wt_m3_per_modules.csv',
    './wt_data/wt_m4_non_modules.csv',
    './wt_data/wt_m4_per_modules.csv',
    './wt_data/wt_m5_non_modules.csv',
    './wt_data/wt_m5_per_modules.csv',
]

conditions_wt = [
    './norm-wt-m1-non.csv',
    './norm-wt-m1-per.csv',
    './norm-wt-m2-non.csv',
    './norm-wt-m2-per.csv',
    './norm-wt-m3-non.csv',
    './norm-wt-m3-per.csv',
    './norm-wt-m4-non.csv',
    './norm-wt-m4-per.csv',
    './norm-wt-m5-non.csv',
    './norm-wt-m5-per.csv',
]

for i in range(synergy_selection):
    synergy_selection = i
    for j in range(len(primtitives_wt_non)):
        sel_primitive_trace(primtitives_wt_non[j], synergy_selection, 'WT Non-Perturbation Synergy {}'.format(i + 1))

    plt.show()

for i in range(synergy_selection):
    synergy_selection = i
    for j in range(len(primtitives_wt_per)):
        sel_primitive_trace(primtitives_wt_per[j], synergy_selection, 'WT Perturbation Synergy {}'.format(i + 1))

    plt.show()


modules_df = df(columns=["Condition", "Perturbation State", "Synergy", "GM", "Ip", "BF", "VL", "St", "TA", "Gs", "Gr"])

module_entry = []

for i in range(0, synergy_selection):
    current_synergy = i + 1
    synergy_tag = "Synergy {}".format(current_synergy)
    for j in range(0, len(conditions_wt)):
        condition_tag = "WT"
        if j == 0 or j % 2 == 0:
            perturbation_state_tag = "Non-Perturbation"
        else:
            perturbation_state_tag = "Perturbation"

        motor_p, motor_m = mp.synergy_extraction(conditions_wt[j], synergy_selection)
        # motor_data = pd.read_csv(modules_wt[j], header=None)
        # motor_m_array = motor_data.to_numpy()
        current_module_set = motor_m[i, :]  # Select all rows for the current column
        identifiers = [condition_tag, perturbation_state_tag, synergy_tag]
        current_module_set = np.ndarray.tolist(current_module_set)
        entry = identifiers + current_module_set
        module_entry.append(entry)

columns = ["Condition", "Perturbation State", "Synergy", "GM", "Ip", "BF", "VL", "St", "TA", "Gs", "Gr"]
muscle_order = ["GM", "Ip", "BF", "VL", "St", "TA", "Gs", "Gr"]
condition_order = ["Non-Perturbation", "Perturbation"]
modules_df = pd.DataFrame.from_records(module_entry, columns=columns)
modules_df_flip = modules_df.T


# annotator = Annotator(syn1, pairs, **plot_params)
# annotator.new_plot(syn1, pairs, plot="barplot", **plot_params)
# annotator.configure(hide_non_significant=False, test="t-test_ind", text_format="star", loc="inside")
# annotator.apply_test().annotate(line_offset_to_group=0.2, line_offset=0.1)
# plt.show()

# %%

# %%

