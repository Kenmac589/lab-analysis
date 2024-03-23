import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame as df
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
        current_primitive = motor_p_full[i * 200 : (i + 1) * 200, synergy_selection - 1]

        primitive_mask = current_primitive > 0.0

        # applying mask to exclude values which were subject to rounding errors
        mcurrent_primitive = np.asarray(current_primitive[primitive_mask])

        # Dealing with local maxima issues at ends of primitives
        # diff_mcurrent = np.diff(mcurrent_primitive_full, axis=0)
        # mcurrent_primitive = mcurrent_primitive_full[np.arange(mcurrent_primitive_full.shape[0]), diff_mcurrent]

        abs_min_ind = np.argmin(mcurrent_primitive)

        # getting maximum
        max_ind = np.argmax(mcurrent_primitive)

        half_width_height = (
            mcurrent_primitive[max_ind] - mcurrent_primitive[abs_min_ind]
        ) / 2

        count_above = np.nonzero(mcurrent_primitive > half_width_height)

        fwhm_index.append(count_above)
        fwhm = np.append(fwhm, [len(count_above[0])])

    fwhm = np.asarray(fwhm)

    return fwhm


def main():
    # Capturing output
    file = open("./output.txt", "w")
    sys.stdout = file

    trial_list = [
        "WT Non Syn 1",
        "WT Non Syn 2",
        "WT Non Syn 3" "WT Per Syn 1",
        "WT Per Syn 2",
        "WT Per Syn 3" "PreDTX Non Syn 1",
        "PreDTX Non Syn 2",
        "PreDTX Non Syn 3",
        "PreDTX Per Syn 1",
        "PreDTX Per Syn 2",
        "PreDTX Per Syn 3",
        "PostDTX Non Syn 1",
        "PostDTX Non Syn 2",
        "PostDTX Non Syn 3",
        "PostDTX Per Syn 1",
        "PostDTX Per Syn 2",
        "PostDTX Per Syn 3",
    ]

    # Import

    # For preDTX primitives
    synergy_selection = 1
    motor_p_data_non = pd.read_csv("./predtx-non-primitives.txt", header=None)
    motor_p_preDTX_non = motor_p_data_non.to_numpy()

    motor_p_data_per = pd.read_csv("./predtx-per-primitives.txt", header=None)
    motor_p_preDTX_per = motor_p_data_per.to_numpy()

    fwhl_non_syn1 = fwhm(motor_p_preDTX_non, synergy_selection)
    # # np.savetxt('./prenon1_widths.csv', fwhl_non_syn1, delimiter=',')

    fwhl_per_syn1 = fwhm(motor_p_preDTX_per, synergy_selection)
    # # np.savetxt('./preper1_widths.csv', fwhl_per_syn1, delimiter=',')

    synergy_selection = 2
    motor_p_data_non = pd.read_csv("./predtx-non-primitives.txt", header=None)
    motor_p_preDTX_non = motor_p_data_non.to_numpy()

    motor_p_data_per = pd.read_csv("./predtx-per-primitives.txt", header=None)
    motor_p_preDTX_per = motor_p_data_per.to_numpy()

    fwhl_non_syn2 = fwhm(motor_p_preDTX_non, synergy_selection)
    # # np.savetxt('./prenon2_widths.csv', fwhl_non_syn2, delimiter=',')

    fwhl_per_syn2 = fwhm(motor_p_preDTX_per, synergy_selection)
    # # np.savetxt('./preper2_widths.csv', fwhl_per_syn2, delimiter=',')

    synergy_selection = 3
    motor_p_data_non = pd.read_csv("./predtx-non-primitives.txt", header=None)
    motor_p_preDTX_non = motor_p_data_non.to_numpy()

    motor_p_data_per = pd.read_csv("./predtx-per-primitives.txt", header=None)
    motor_p_preDTX_per = motor_p_data_per.to_numpy()

    fwhl_non_syn3 = fwhm(motor_p_preDTX_non, synergy_selection)
    # np.savetxt('./prenon3_widths.csv', fwhl_non_syn3, delimiter=',')

    fwhl_per_syn3 = fwhm(motor_p_preDTX_per, synergy_selection)
    # np.savetxt('./preper3_widths.csv', fwhl_per_syn3, delimiter=',')

    motor_p_pre_non_df = pd.read_csv("./predtx-non-primitives.txt", header=None)
    motor_p_pre_per_df = pd.read_csv("./predtx-per-primitives.txt", header=None)

    fwhl_pre_non = dict()
    fwhl_pre_per = dict()

    # For PostDTX Conditions
    synergy_selection = 1
    motor_p_data_non_post = pd.read_csv("./postdtx-non-primitives.txt", header=0)
    motor_p_preDTX_non_post = motor_p_data_non_post.to_numpy()

    motor_p_data_per_post = pd.read_csv("./postdtx-per-primitives.txt", header=0)
    motor_p_preDTX_per_post = motor_p_data_per_post.to_numpy()

    # fwhl_non_post, fwhl_non_post_start_stop, fwhl_height_non_post = full_width_half_abs_min(motor_p_preDTX_non_post, synergy_selection)
    fwhl_post_non_syn1 = fwhm(motor_p_preDTX_non_post, synergy_selection)

    # fwhl_per_post, fwhl_per_post_start_stop, fwhl_height_per_post = full_width_half_abs_min(motor_p_preDTX_per_post, synergy_selection)
    fwhl_post_per_syn1 = fwhm(motor_p_preDTX_per_post, synergy_selection)

    synergy_selection = 2
    motor_p_data_non_post = pd.read_csv("./postdtx-non-primitives.txt", header=0)
    motor_p_preDTX_non_post = motor_p_data_non_post.to_numpy()

    motor_p_data_per_post = pd.read_csv("./postdtx-per-primitives.txt", header=0)
    motor_p_preDTX_per_post = motor_p_data_per_post.to_numpy()

    # fwhl_non_post, fwhl_non_post_start_stop, fwhl_height_non_post = full_width_half_abs_min(motor_p_preDTX_non_post, synergy_selection)
    fwhl_post_non_syn2 = fwhm(motor_p_preDTX_non_post, synergy_selection)

    # fwhl_per_post, fwhl_per_post_start_stop, fwhl_height_per_post = full_width_half_abs_min(motor_p_preDTX_per_post, synergy_selection)
    fwhl_post_per_syn2 = fwhm(motor_p_preDTX_per_post, synergy_selection)

    synergy_selection = 3
    motor_p_data_non_post = pd.read_csv("./postdtx-non-primitives.txt", header=0)
    motor_p_preDTX_non_post = motor_p_data_non_post.to_numpy()

    motor_p_data_per_post = pd.read_csv("./postdtx-per-primitives.txt", header=0)
    motor_p_preDTX_per_post = motor_p_data_per_post.to_numpy()

    # fwhl_non_post, fwhl_non_post_start_stop, fwhl_height_non_post = full_width_half_abs_min(motor_p_preDTX_non_post, synergy_selection)
    fwhl_post_non_syn3 = fwhm(motor_p_preDTX_non_post, synergy_selection)

    # fwhl_per_post, fwhl_per_post_start_stop, fwhl_height_per_post = full_width_half_abs_min(motor_p_preDTX_per_post, synergy_selection)
    fwhl_post_per_syn3 = fwhm(motor_p_preDTX_per_post, synergy_selection)

    # For WT
    synergy_selection = 1
    motor_p_data_non = pd.read_csv("./CoM-M1/primitives-com-m1-non.csv", header=None)
    motor_p_wt_non = motor_p_data_non.to_numpy()

    motor_p_data_per = pd.read_csv("./CoM-M1/primitives-com-m1-per.csv", header=None)
    motor_p_wt_per = motor_p_data_per.to_numpy()

    fwhl_wt_non_syn1 = fwhm(motor_p_preDTX_non, synergy_selection)
    # # np.savetxt('./prenon1_widths.csv', fwhl_non_syn1, delimiter=',')

    fwhl_wt_per_syn1 = fwhm(motor_p_preDTX_per, synergy_selection)
    # # np.savetxt('./preper1_widths.csv', fwhl_per_syn1, delimiter=',')

    synergy_selection = 2
    motor_p_data_non = pd.read_csv("./CoM-M1/primitives-com-m1-non.csv", header=None)
    motor_p_wt_non = motor_p_data_non.to_numpy()

    motor_p_data_per = pd.read_csv("./CoM-M1/primitives-com-m1-per.csv", header=None)
    motor_p_wt_per = motor_p_data_per.to_numpy()

    fwhl_wt_non_syn2 = fwhm(motor_p_preDTX_non, synergy_selection)
    # # np.savetxt('./prenon2_widths.csv', fwhl_non_syn2, delimiter=',')

    fwhl_wt_per_syn2 = fwhm(motor_p_preDTX_per, synergy_selection)
    # # np.savetxt('./preper2_widths.csv', fwhl_per_syn2, delimiter=',')

    synergy_selection = 3
    motor_p_data_non = pd.read_csv("./CoM-M1/primitives-com-m1-non.csv", header=None)
    motor_p_wt_non = motor_p_data_non.to_numpy()

    motor_p_data_per = pd.read_csv("./CoM-M1/primitives-com-m1-per.csv", header=None)
    motor_p_wt_per = motor_p_data_per.to_numpy()

    fwhl_wt_non_syn3 = fwhm(motor_p_preDTX_non, synergy_selection)
    # np.savetxt('./prenon3_widths.csv', fwhl_non_syn3, delimiter=',')

    fwhl_wt_per_syn3 = fwhm(motor_p_preDTX_per, synergy_selection)
    # Analysis of fwhl_lenghts

    # Results dataframe
    fwhm_df = df()

    fwhm_df = pd.concat([fwhm_df, df({("WT Non Syn 1"): fwhl_wt_non_syn1})], axis=1)
    fwhm_df = pd.concat([fwhm_df, df({("WT Non Syn 2"): fwhl_wt_non_syn2})], axis=1)
    fwhm_df = pd.concat([fwhm_df, df({("WT Non Syn 3"): fwhl_wt_non_syn3})], axis=1)
    fwhm_df = pd.concat([fwhm_df, df({("WT Per Syn 1"): fwhl_wt_per_syn1})], axis=1)
    fwhm_df = pd.concat([fwhm_df, df({("WT Per Syn 2"): fwhl_wt_per_syn2})], axis=1)
    fwhm_df = pd.concat([fwhm_df, df({("WT Per Syn 3"): fwhl_wt_per_syn3})], axis=1)
    fwhm_df = pd.concat([fwhm_df, df({("PreDTX Non Syn 1"): fwhl_non_syn1})], axis=1)
    fwhm_df = pd.concat([fwhm_df, df({("PreDTX Non Syn 2"): fwhl_non_syn2})], axis=1)
    fwhm_df = pd.concat([fwhm_df, df({("PreDTX Non Syn 3"): fwhl_non_syn3})], axis=1)
    fwhm_df = pd.concat([fwhm_df, df({("PreDTX Per Syn 1"): fwhl_per_syn1})], axis=1)
    fwhm_df = pd.concat([fwhm_df, df({("PreDTX Per Syn 2"): fwhl_per_syn2})], axis=1)
    fwhm_df = pd.concat([fwhm_df, df({("PreDTX Per Syn 3"): fwhl_per_syn3})], axis=1)
    fwhm_df = pd.concat(
        [fwhm_df, df({("PostDTX Non Syn 1"): fwhl_post_non_syn1})], axis=1
    )
    fwhm_df = pd.concat(
        [fwhm_df, df({("PostDTX Non Syn 2"): fwhl_post_non_syn2})], axis=1
    )
    fwhm_df = pd.concat(
        [fwhm_df, df({("PostDTX Non Syn 3"): fwhl_post_non_syn3})], axis=1
    )
    fwhm_df = pd.concat(
        [fwhm_df, df({("PostDTX Per Syn 1"): fwhl_post_per_syn1})], axis=1
    )
    fwhm_df = pd.concat(
        [fwhm_df, df({("PostDTX Per Syn 2"): fwhl_post_per_syn2})], axis=1
    )
    fwhm_df = pd.concat(
        [fwhm_df, df({("PostDTX Per Syn 3"): fwhl_post_per_syn3})], axis=1
    )

    # Clustering based on synergy
    fwhm_syn1 = fwhm_df.loc[:, [col for col in fwhm_df.columns if "Syn 1" in col]]
    fwhm_syn2 = fwhm_df.loc[:, [col for col in fwhm_df.columns if "Syn 2" in col]]
    fwhm_syn3 = fwhm_df.loc[:, [col for col in fwhm_df.columns if "Syn 3" in col]]

    pairs_syn1_ttest = [
        ("WT Non Syn 1", "WT Per Syn 1"),
        ("WT Non Syn 1", "PreDTX Non Syn 1"),
        ("PreDTX Per Syn 1", "PreDTX Non Syn 1"),
        ("PreDTX Per Syn 1", "PostDTX Non Syn 1"),
        ("PreDTX Per Syn 1", "PostDTX Per Syn 1"),
        ("PostDTX Non Syn 1", "PreDTX Non Syn 1"),
    ]
    pairs_syn2_ttest = [
        ("WT Non Syn 2", "WT Per Syn 2"),
        ("WT Non Syn 2", "PreDTX Non Syn 2"),
        ("PreDTX Per Syn 2", "PreDTX Non Syn 2"),
        ("PreDTX Per Syn 2", "PostDTX Non Syn 2"),
        ("PreDTX Per Syn 2", "PostDTX Per Syn 2"),
        ("PostDTX Non Syn 2", "PreDTX Non Syn 2"),
    ]
    pairs_syn3_ttest = [
        ("WT Non Syn 3", "WT Per Syn 3"),
        ("WT Non Syn 3", "PreDTX Non Syn 3"),
        ("PreDTX Per Syn 3", "PostDTX Non Syn 3"),
        ("PreDTX Per Syn 3", "PostDTX Non Syn 3"),
        ("PreDTX Per Syn 3", "PostDTX Per Syn 3"),
        ("PostDTX Non Syn 3", "PreDTX Non Syn 3"),
    ]

    # Plotting
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set(style="white", rc=custom_params)

    plt.title("Full Width Half Length for Synergy 1")
    plt.ylim(0, 250)
    syn1 = sns.barplot(
        x=fwhm_syn1.columns, y=fwhm_syn1.mean(), order=fwhm_syn1.columns, zorder=2
    )
    syn1.errorbar(
        x=fwhm_syn1.columns,
        y=fwhm_syn1.mean(),
        yerr=fwhm_syn1.std(),
        capsize=3,
        fmt="none",
        c="k",
        zorder=1,
    )
    annotator = Annotator(syn1, pairs_syn1_ttest, data=fwhm_syn1)
    annotator.configure(
        hide_non_significant=True, test="t-test_ind", text_format="simple", loc="inside"
    )
    annotator.apply_test().annotate(
        line_offset_to_group=0.2, line_offset=0.1
    )  # When keeping inside
    # annotator.apply_test().annotate()
    plt.show()

    plt.title("Full Width Half Length for Synergy 2")
    plt.ylim(0, 250)
    syn2 = sns.barplot(
        x=fwhm_syn2.columns, y=fwhm_syn2.mean(), order=fwhm_syn2.columns, zorder=2
    )
    syn2.errorbar(
        x=fwhm_syn2.columns,
        y=fwhm_syn2.mean(),
        yerr=fwhm_syn2.std(),
        capsize=3,
        fmt="none",
        c="k",
        zorder=1,
    )
    annotator = Annotator(syn2, pairs_syn2_ttest, data=fwhm_syn2)
    annotator.configure(
        hide_non_significant=True,
        test="t-test_welch",
        text_format="simple",
        loc="inside",
    )
    annotator.apply_test().annotate(
        line_offset_to_group=0.3, line_offset=0.1
    )  # when inside
    # annotator.apply_test().annotate()
    plt.show()

    plt.title("Full Width Half Length for Synergy 3")
    plt.ylim(0, 250)
    syn3 = sns.barplot(
        x=fwhm_syn3.columns, y=fwhm_syn3.mean(), order=fwhm_syn3.columns, zorder=2
    )
    syn3.errorbar(
        x=fwhm_syn3.columns,
        y=fwhm_syn3.mean(),
        yerr=fwhm_syn3.std(),
        capsize=3,
        fmt="none",
        c="k",
        zorder=1,
    )
    annotator = Annotator(syn3, pairs_syn3_ttest, data=fwhm_syn3)
    annotator.configure(
        hide_non_significant=True,
        test="t-test_welch",
        text_format="simple",
        loc="inside",
    )
    annotator.apply_test().annotate(line_offset_to_group=0.3, line_offset=0.1)
    plt.show()

    # Statistics
    print("Stats for Synergy 1")
    print("-------------------")
    print()
    print("For Non-Perturbation")
    print(st.ttest_ind(fwhm_syn1["PreDTX Non Syn 1"], fwhm_syn1["PostDTX Non Syn 1"]))
    print()
    print("For Perturbation")
    print(st.ttest_ind(fwhm_syn1["PreDTX Per Syn 1"], fwhm_syn1["PostDTX Per Syn 1"]))
    print()
    print("Stats for Synergy 2")
    print("-------------------")
    print()
    print("For Non-Perturbation")
    print(st.ttest_ind(fwhm_syn2["PreDTX Non Syn 2"], fwhm_syn2["PostDTX Non Syn 2"]))
    print()
    print("For Perturbation")
    print(st.ttest_ind(fwhm_syn2["PreDTX Per Syn 2"], fwhm_syn2["PostDTX Per Syn 2"]))
    print()
    print("Stats for Synergy 3")
    print("-------------------")
    print()
    print("For Non-Perturbation")
    print(st.ttest_ind(fwhm_syn3["PreDTX Non Syn 3"], fwhm_syn3["PostDTX Non Syn 3"]))
    print()
    print("For Perturbation")
    print(st.ttest_ind(fwhm_syn3["PreDTX Per Syn 3"], fwhm_syn3["PostDTX Per Syn 3"]))

    # Save output to txt file
    file.close()


if __name__ == "__main__":
    main()
