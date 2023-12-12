import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame as df
import seaborn as sns
from scipy import stats as st
from statannotations.Annotator import Annotator

def fwhm(motor_p_full, synergy_selection):
    """full width half maxiumum calculation
    @param: motor_p_full: full length numpy array of selected motor
    primitives

    @return: mean_fwhm: Mean value for the width of the primitives
    """

    number_cycles = len(motor_p_full) // 200

    # Save
    fwhm = np.array([])
    fwhm_index = [[]]

    for i in range(number_cycles):
        current_primitive = motor_p_full[i * 200: (i + 1) * 200, synergy_selection - 1]

        primitive_mask = current_primitive > 0.0

        # applying mask to exclude values which were subject to rounding errors
        mcurrent_primitive = np.asarray(current_primitive[primitive_mask])

        # Dealing with local maxima issues at ends of primitives
        # diff_mcurrent = np.diff(mcurrent_primitive_full, axis=0)
        # mcurrent_primitive = mcurrent_primitive_full[np.arange(mcurrent_primitive_full.shape[0]), diff_mcurrent]

        abs_min_ind = np.argmin(mcurrent_primitive)

        # getting maximum
        max_ind = np.argmax(mcurrent_primitive)


        half_width_height = (mcurrent_primitive[max_ind] - mcurrent_primitive[abs_min_ind]) / 2

        count_above = np.nonzero(mcurrent_primitive > half_width_height)

        fwhm_index.append(count_above)
        fwhm = np.append(fwhm, [len(count_above[0])])

    # fwhm = np.asarray(fwhm)

    return fwhm

def main():
    # Capturing output
    conditions = [
        './CoM-M1/primitives-com-m1-non.csv',
        './predtx-non-primitives-test.txt',
        './postdtx-non-primitives.txt',
        './CoM-M1/primitives-com-m1-per.csv',
        './predtx-per-primitives-test.txt',
        './postdtx-per-primitives.txt',
    ]

    conditions_name = ["WT", "PreDTX", "PostDTX", "WT", "PreDTX", "PostDTX"]
    perturbation_state = ["Non-Perturbation", "Non-Perturbation", "Non-Perturbation", "Perturbation", "Perturbation", "Perturbation"]
    perturbation_state_order = ["Non-Perturbation", "Perturbation"]

    synergy_selection = 3

    fwhm_df = df(columns=["Condition", "Perturbation State", "Synergy", "FWHM"])

    for i in range(0, synergy_selection):
        current_synergy = i + 1
        synergy_tag = "Synergy {}".format(current_synergy)
        print(synergy_tag)
        for j in range(0, len(conditions)):
            condition_tag = conditions_name[j]
            perturbation_state_tag = perturbation_state[j]
            # print(condition_tag + " " + perturbation_state_tag)
            motor_p_data = pd.read_csv(conditions[j], header=0)
            motor_p_array = motor_p_data.to_numpy()
            fwhm_list = fwhm(motor_p_array, current_synergy)
            for k in range(0, len(fwhm_list)):
                fwhm_entry = [[condition_tag, perturbation_state_tag, synergy_tag, fwhm_list[k]]]
                fwhm_df = fwhm_df._append(pd.DataFrame(fwhm_entry, columns=["Condition", "Perturbation State", "Synergy", "FWHM"]), ignore_index=True)

    print(fwhm_df)
    syn1_df = fwhm_df[fwhm_df["Synergy"] == "Synergy 1"]
    syn2_df = fwhm_df[fwhm_df["Synergy"] == "Synergy 2"]
    syn3_df = fwhm_df[fwhm_df["Synergy"] == "Synergy 3"]

    # Plotting
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set(style="white", font_scale=1.5, rc=custom_params)

    pairs = [
        [("WT", "Non-Perturbation"), ("WT", "Perturbation")],
        [("PreDTX", "Non-Perturbation"), ("PreDTX", "Perturbation")],
        [("PostDTX", "Non-Perturbation"), ("PostDTX", "Perturbation")],
    ]

    # Plot for Synergy 1
    plot_params = {
        "data": syn1_df,
        "x": "Condition",
        "y": "FWHM",
        "hue": "Perturbation State",
        "hue_order": perturbation_state_order,
    }

    plt.title("FWHM For Synergy 1")
    plt.ylim(0, 1)
    plt.ylim(0, 250)
    syn1 = sns.barplot(**plot_params, ci="sd", capsize=0.05)
    plt.ylabel('')
    plt.legend(loc='best', fontsize=12)
    annotator = Annotator(syn1, pairs, **plot_params)
    annotator.new_plot(syn1, pairs, plot="barplot", **plot_params)
    annotator.configure(hide_non_significant=False, test="t-test_ind", text_format="star", loc="inside")
    annotator.apply_test().annotate(line_offset_to_group=0.2, line_offset=0.1)
    plt.show()

    # Plot for Synergy 2
    plot_params = {
        "data": syn2_df,
        "x": "Condition",
        "y": "FWHM",
        "hue": "Perturbation State",
        "hue_order": perturbation_state_order,
    }

    plt.title("FWHM For Synergy 2")
    plt.ylim(0, 250)
    syn2 = sns.barplot(**plot_params, ci="sd", capsize=0.05)
    plt.ylabel('')
    plt.legend(loc='best', fontsize=12)
    annotator = Annotator(syn2, pairs, **plot_params)
    annotator.new_plot(syn2, pairs, plot="barplot", **plot_params)
    annotator.configure(hide_non_significant=False, test="t-test_ind", text_format="star", loc="inside")
    annotator.apply_test().annotate(line_offset_to_group=0.2, line_offset=0.1)
    plt.show()

    # Plot for Synergy 3
    plot_params = {
        "data": syn3_df,
        "x": "Condition",
        "y": "FWHM",
        "hue": "Perturbation State",
        "hue_order": perturbation_state_order,
    }

    plt.title("FWHM For Synergy 3")
    plt.ylim(0, 250)
    syn3 = sns.barplot(**plot_params, ci="sd", capsize=0.05)
    plt.ylabel('')
    plt.legend(loc='best', fontsize=12)
    annotator = Annotator(syn3, pairs, **plot_params)
    annotator.new_plot(syn3, pairs, plot="barplot", **plot_params)
    annotator.configure(hide_non_significant=False, test="t-test_ind", text_format="star", loc="inside")
    annotator.apply_test().annotate(line_offset_to_group=0.2, line_offset=0.1)
    plt.show()

    # Figure Attempt

if __name__ == "__main__":
    main()
