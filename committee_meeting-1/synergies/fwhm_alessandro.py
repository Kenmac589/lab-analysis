import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
from pandas import DataFrame as df
import seaborn as sns
from scipy import signal
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

    fwhm = np.asarray(fwhm)

    return fwhm


def main():

    # For preDTX primitives
    synergy_selection = 1
    motor_p_data_non = pd.read_csv('./predtx-non-primitives.txt', header=None)
    motor_p_preDTX_non = motor_p_data_non.to_numpy()

    motor_p_data_per = pd.read_csv('./predtx-per-primitives.txt', header=None)
    motor_p_preDTX_per = motor_p_data_per.to_numpy()

    fwhl_non_syn1 = fwhm(motor_p_preDTX_non, synergy_selection)
    # # np.savetxt('./prenon1_widths.csv', fwhl_non_syn1, delimiter=',')

    fwhl_per_syn1 = fwhm(motor_p_preDTX_per, synergy_selection)
    # # np.savetxt('./preper1_widths.csv', fwhl_per_syn1, delimiter=',')

    synergy_selection = 2
    motor_p_data_non = pd.read_csv('./predtx-non-primitives.txt', header=None)
    motor_p_preDTX_non = motor_p_data_non.to_numpy()

    motor_p_data_per = pd.read_csv('./predtx-per-primitives.txt', header=None)
    motor_p_preDTX_per = motor_p_data_per.to_numpy()

    fwhl_non_syn2 = fwhm(motor_p_preDTX_non, synergy_selection)
    # # np.savetxt('./prenon2_widths.csv', fwhl_non_syn2, delimiter=',')

    fwhl_per_syn2 = fwhm(motor_p_preDTX_per, synergy_selection)
    # # np.savetxt('./preper2_widths.csv', fwhl_per_syn2, delimiter=',')

    synergy_selection = 3
    motor_p_data_non = pd.read_csv('./predtx-non-primitives.txt', header=None)
    motor_p_preDTX_non = motor_p_data_non.to_numpy()

    motor_p_data_per = pd.read_csv('./predtx-per-primitives.txt', header=None)
    motor_p_preDTX_per = motor_p_data_per.to_numpy()

    fwhl_non_syn3 = fwhm(motor_p_preDTX_non, synergy_selection)
    # np.savetxt('./prenon3_widths.csv', fwhl_non_syn3, delimiter=',')

    fwhl_per_syn3 = fwhm(motor_p_preDTX_per, synergy_selection)
    # np.savetxt('./preper3_widths.csv', fwhl_per_syn3, delimiter=',')

    # For PostDTX Conditions
    synergy_selection = 1
    motor_p_data_non_post = pd.read_csv('./postdtx-non-primitives.txt', header=None)
    motor_p_preDTX_non_post = motor_p_data_non_post.to_numpy()

    motor_p_data_per_post = pd.read_csv('./postdtx-per-primitives.txt', header=None)
    motor_p_preDTX_per_post = motor_p_data_per_post.to_numpy()

    # fwhl_non_post, fwhl_non_post_start_stop, fwhl_height_non_post = full_width_half_abs_min(motor_p_preDTX_non_post, synergy_selection)
    fwhl_post_non_syn1 = fwhm(motor_p_preDTX_non_post, synergy_selection)

    # fwhl_per_post, fwhl_per_post_start_stop, fwhl_height_per_post = full_width_half_abs_min(motor_p_preDTX_per_post, synergy_selection)
    fwhl_post_per_syn1 = fwhm(motor_p_preDTX_per_post, synergy_selection)

    synergy_selection = 2
    motor_p_data_non_post = pd.read_csv('./postdtx-non-primitives.txt', header=None)
    motor_p_preDTX_non_post = motor_p_data_non_post.to_numpy()

    motor_p_data_per_post = pd.read_csv('./postdtx-per-primitives.txt', header=None)
    motor_p_preDTX_per_post = motor_p_data_per_post.to_numpy()

    # fwhl_non_post, fwhl_non_post_start_stop, fwhl_height_non_post = full_width_half_abs_min(motor_p_preDTX_non_post, synergy_selection)
    fwhl_post_non_syn2 = fwhm(motor_p_preDTX_non_post, synergy_selection)

    # fwhl_per_post, fwhl_per_post_start_stop, fwhl_height_per_post = full_width_half_abs_min(motor_p_preDTX_per_post, synergy_selection)
    fwhl_post_per_syn2 = fwhm(motor_p_preDTX_per_post, synergy_selection)

    synergy_selection = 3
    motor_p_data_non_post = pd.read_csv('./postdtx-non-primitives.txt', header=None)
    motor_p_preDTX_non_post = motor_p_data_non_post.to_numpy()

    motor_p_data_per_post = pd.read_csv('./postdtx-per-primitives.txt', header=None)
    motor_p_preDTX_per_post = motor_p_data_per_post.to_numpy()

    # fwhl_non_post, fwhl_non_post_start_stop, fwhl_height_non_post = full_width_half_abs_min(motor_p_preDTX_non_post, synergy_selection)
    fwhl_post_non_syn3 = fwhm(motor_p_preDTX_non_post, synergy_selection)

    # fwhl_per_post, fwhl_per_post_start_stop, fwhl_height_per_post = full_width_half_abs_min(motor_p_preDTX_per_post, synergy_selection)
    fwhl_post_per_syn3 = fwhm(motor_p_preDTX_per_post, synergy_selection)

    # Analysis of fwhl_lenghts

    # Results dataframe
    fwhm_df = df()
    fwhm_df = pd.concat([fwhm_df, df({('PreDTX Non Syn 1'): fwhl_non_syn1})], axis=1)
    fwhm_df = pd.concat([fwhm_df, df({('PreDTX Non Syn 2'): fwhl_non_syn2})], axis=1)
    fwhm_df = pd.concat([fwhm_df, df({('PreDTX Non Syn 3'): fwhl_non_syn3})], axis=1)
    fwhm_df = pd.concat([fwhm_df, df({('PreDTX Per Syn 1'): fwhl_per_syn1})], axis=1)
    fwhm_df = pd.concat([fwhm_df, df({('PreDTX Per Syn 2'): fwhl_per_syn1})], axis=1)
    fwhm_df = pd.concat([fwhm_df, df({('PreDTX Per Syn 3'): fwhl_non_syn1})], axis=1)
    fwhm_df = pd.concat([fwhm_df, df({('PostDTX Non Syn 1'): fwhl_post_non_syn1})], axis=1)
    fwhm_df = pd.concat([fwhm_df, df({('PostDTX Non Syn 2'): fwhl_post_non_syn2})], axis=1)
    fwhm_df = pd.concat([fwhm_df, df({('PostDTX Non Syn 3'): fwhl_post_non_syn3})], axis=1)
    fwhm_df = pd.concat([fwhm_df, df({('PostDTX Per Syn 1'): fwhl_post_per_syn1})], axis=1)
    fwhm_df = pd.concat([fwhm_df, df({('PostDTX Per Syn 2'): fwhl_post_per_syn2})], axis=1)
    fwhm_df = pd.concat([fwhm_df, df({('PostDTX Per Syn 3'): fwhl_post_per_syn3})], axis=1)

    # Clustering based on synergy
    fwhm_syn1 = fwhm_df.loc[:, [col for col in fwhm_df.columns if 'Syn 1' in col]]
    fwhm_syn2 = fwhm_df.loc[:, [col for col in fwhm_df.columns if 'Syn 2' in col]]
    fwhm_syn3 = fwhm_df.loc[:, [col for col in fwhm_df.columns if 'Syn 3' in col]]

    pairs_syn1 = [("PreDTX Per Syn 1", "PreDTX Non Syn 1"), ("PreDTX Per Syn 1", "PostDTX Non Syn 1"), ("PreDTX Per Syn 1", "PostDTX Per Syn 1"), ("PostDTX Non Syn 1", "PreDTX Non Syn 1")]
    pairs_syn2 = [("PreDTX Per Syn 2", "PreDTX Non Syn 2"), ("PreDTX Per Syn 2", "PostDTX Non Syn 2"), ("PreDTX Per Syn 2", "PostDTX Per Syn 2"), ("PostDTX Non Syn 2", "PreDTX Non Syn 2")]
    pairs_syn3 = [("PreDTX Per Syn 3", "PostDTX Non Syn 3"), ("PreDTX Per Syn 3", "PostDTX Non Syn 3"), ("PreDTX Per Syn 3", "PostDTX Per Syn 3"), ("PostDTX Non Syn 3", "PreDTX Non Syn 3")]

    # Plotting
    sns.set_style("whitegrid")

    plt.title("Full Width Half Length for Synergy 1")
    plt.ylim(0, 200)
    ax = sns.barplot(x=fwhm_syn1.columns, y=fwhm_syn1.mean(), order=fwhm_syn1.columns)
    ax.errorbar(x=fwhm_syn1.columns, y=fwhm_syn1.mean(), yerr=fwhm_syn1.std(), capsize=2, fmt="none", c="k")
    annotator = Annotator(ax, pairs_syn1, data=fwhm_syn1)
    annotator.configure(test='Mann-Whitney', text_format='star')
    annotator.apply_and_annotate()
    plt.show()

    plt.title("Full Width Half Length for Synergy 2")
    plt.ylim(0, 200)
    ax = sns.barplot(x=fwhm_syn2.columns, y=fwhm_syn2.mean(), order=fwhm_syn2.columns)
    ax.errorbar(x=fwhm_syn2.columns, y=fwhm_syn2.mean(), yerr=fwhm_syn2.std(), capsize=2, fmt="none", c="k")
    annotator = Annotator(ax, pairs_syn2, data=fwhm_syn2)
    annotator.configure(test='Mann-Whitney', text_format='star')
    annotator.apply_and_annotate()
    plt.show()

    plt.title("Full Width Half Length for Synergy 3")
    plt.ylim(0, 200)
    ax = sns.barplot(x=fwhm_syn3.columns, y=fwhm_syn3.mean(), order=fwhm_syn3.columns)
    ax.errorbar(x=fwhm_syn3.columns, y=fwhm_syn3.mean(), yerr=fwhm_syn3.std(), capsize=2, fmt="none", c="k")
    annotator = Annotator(ax, pairs_syn3, data=fwhm_syn3)
    annotator.configure(test='Mann-Whitney', text_format='star')
    annotator.apply_and_annotate()
    plt.show()

    # Results Dictionnary
    results = dict()
    results.update({'PreDTX Non Syn 1': [np.mean(fwhl_non_syn1), np.std(fwhl_non_syn1)]})
    results.update({'PreDTX Non Syn 2': [np.mean(fwhl_non_syn2), np.std(fwhl_non_syn2)]})
    results.update({'PreDTX Non Syn 3': [np.mean(fwhl_non_syn3), np.std(fwhl_non_syn3)]})
    results.update({'PreDTX Per Syn 1': [np.mean(fwhl_per_syn1), np.std(fwhl_per_syn1)]})
    results.update({'PreDTX Per Syn 2': [np.mean(fwhl_per_syn2), np.std(fwhl_per_syn2)]})
    results.update({'PreDTX Per Syn 3': [np.mean(fwhl_per_syn3), np.std(fwhl_per_syn3)]})
    results.update({'PostDTX Non Syn 1': [np.mean(fwhl_post_non_syn1), np.std(fwhl_post_non_syn1)]})
    results.update({'PostDTX Non Syn 2': [np.mean(fwhl_post_non_syn2), np.std(fwhl_post_non_syn2)]})
    results.update({'PostDTX Non Syn 3': [np.mean(fwhl_post_non_syn3), np.std(fwhl_post_non_syn3)]})
    results.update({'PostDTX Per Syn 1': [np.mean(fwhl_post_per_syn1), np.std(fwhl_post_per_syn1)]})
    results.update({'PostDTX Per Syn 2': [np.mean(fwhl_post_per_syn2), np.std(fwhl_post_per_syn2)]})
    results.update({'PostDTX Per Syn 3': [np.mean(fwhl_post_per_syn3), np.std(fwhl_post_per_syn3)]})

    # Moving

    # Synergy based comparison
    # res_syn1 = {key: results[key] for key in results.keys() & {'PreDTX Non Syn 1', 'PreDTX Per Syn 1', 'PostDTX Non Syn 1', 'PostDTX Per Syn 1'}}
    # res_syn2 = {key: results[key] for key in results.keys() & {'PreDTX Non Syn 2', 'PreDTX Per Syn 2', 'PostDTX Non Syn 2', 'PostDTX Per Syn 2'}}
    # res_syn3 = {key: results[key] for key in results.keys() & {'PreDTX Non Syn 3', 'PreDTX Per Syn 3', 'PostDTX Non Syn 3', 'PostDTX Per Syn 3'}}
    res_syn1 = dict((sel, results[sel]) for sel in ['PreDTX Non Syn 1', 'PreDTX Per Syn 1', 'PostDTX Non Syn 1', 'PostDTX Per Syn 1'] if sel in results)
    res_syn2 = dict((sel, results[sel]) for sel in ['PreDTX Non Syn 2', 'PreDTX Per Syn 2', 'PostDTX Non Syn 2', 'PostDTX Per Syn 2'] if sel in results)
    res_syn3 = dict((sel, results[sel]) for sel in ['PreDTX Non Syn 3', 'PreDTX Per Syn 3', 'PostDTX Non Syn 3', 'PostDTX Per Syn 3'] if sel in results)

    # For Synergy 1
    trials_one = list(res_syn1.keys())
    mean_fwhl_one = [value[0] for value in res_syn1.values()]
    std_fwhl_one = [value[1] for value in res_syn1.values()]
    # print(res_syn1)

    # For Synergy 2
    trials_two = list(res_syn2.keys())
    mean_fwhl_two = [value[0] for value in res_syn2.values()]
    std_fwhl_two = [value[1] for value in res_syn2.values()]
    # print(res_syn2)

    # For Synergy 3
    trials_three = list(res_syn3.keys())
    mean_fwhl_three = [value[0] for value in res_syn3.values()]
    std_fwhl_three = [value[1] for value in res_syn3.values()]
    # print(res_syn3)

    # Plotting

    plt.title("Full Width Half Length for Synergy 1")
    plt.ylim(0, 200)
    ax = sns.barplot(x=trials_one, y=mean_fwhl_one)
    ax.errorbar(x=trials_one, y=mean_fwhl_one, yerr=std_fwhl_one, capsize=2, lolims=True, fmt="none", c="k")
    plt.tight_layout()
    plt.show()

    plt.title("Full Width Half Length for Synergy 2")
    plt.ylim(0, 200)
    ax = sns.barplot(x=trials_two, y=mean_fwhl_two)
    ax.errorbar(x=trials_two, y=mean_fwhl_two, yerr=std_fwhl_two, capsize=2, fmt="none", c="k")
    plt.tight_layout()

    plt.title("Full Width Half Length for Synergy 3")
    plt.ylim(0, 200)
    ax = sns.barplot(x=trials_three, y=mean_fwhl_three)
    ax.errorbar(x=trials_three, y=mean_fwhl_three, yerr=std_fwhl_three, capsize=2, fmt="none", c="k")
    plt.tight_layout()

    # Save output to txt file
    sys.stdout = open("./log.txt", "a")
    sys.stdout.close()

if __name__ == "__main__":
    main()
