"""Non-Negative Matrix Factorization for Muscle Synergy Extraction
This program performs Non-Negative Matrix Factorization for determing
the appropriate number of components/muscle channels to use for muscle
synergy extraction.

Some functions could be more recursive however, they have been used in
applications such as synergy selection.
"""
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
from scipy import ndimage, signal, interpolate
from sklearn.decomposition import NMF

def nnmf_factorize(A, k):
    """Non-Negative Matrix Factorization for Muscle Synergy Extraction
    @param A: input matrix
    @param k: number of components (muscle channels)

    @return W: motor primitives
    @return H: motor modules
    @return C: factorized matrix
    """
    nmf = NMF(n_components=k, init='random', random_state=0)
    W = nmf.fit_transform(A)
    H = nmf.components_
    C = np.dot(W, H)
    return W, H, C

def synergy_extraction(data_input, synergy_selection):
    """Synergy Extraction from factorized matricies
    @param data_input: path to csv data file
    @param synergy_selection:

    @return W: motor primitives
    @return H: motor modules
    """

    # Load Data
    data = pd.read_csv(data_input, header=None)
    A = data.to_numpy()

    # Choosing best number of components
    chosen_synergies = synergy_selection
    W, H, C = nnmf_factorize(A, chosen_synergies)

    motor_modules = H
    motor_primitives = W

    return motor_primitives, motor_modules

def read_all_csv(directory_path):
    data_dict = {}  # Initialize an empty dictionary to store the data

    if not os.path.isdir(directory_path):
        print(f"{directory_path} is not a valid directory.")
        return

    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            data = pd.read_csv(file_path)
            data_dict[filename] = data

    return data_dict


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
    half_width_values = []

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
        half_width_values.append(half_width_height)

        count_above = np.nonzero(mcurrent_primitive > half_width_height)

        fwhm_index.append(count_above)
        fwhm = np.append(fwhm, [len(count_above[0])])
    fwhm = np.asarray(fwhm)

    return fwhm, half_width_values

def full_width_half_first_min(motor_p_full, synergy_selection):
    """Full width half maxiumum calculation
    @param: motor_p_full: full length numpy array of selected motor
    primitives

    @return: mean_fwhm: Mean value for the width of the primitives
    """

    number_cycles = len(motor_p_full) // 200

    # Save
    fwhl = np.array([])
    half_width_height_array = np.array([])
    fwhl_start_stop = np.empty((number_cycles, 0))

    for i in range(number_cycles):
        current_primitive = motor_p_full[i * 200: (i + 1) * 200, synergy_selection - 2]

        primitive_mask = current_primitive > 0.0

        # applying mask to exclude values which were subject to rounding errors
        mcurrent_primitive = np.asarray(current_primitive[primitive_mask])

        # Dealing with local maxima issues at ends of primitives
        # diff_mcurrent = np.diff(mcurrent_primitive_full, axis=0)
        # mcurrent_primitive = mcurrent_primitive_full[np.arange(mcurrent_primitive_full.shape[0]), diff_mcurrent]

        abs_min_ind = np.argmin(mcurrent_primitive)

        # getting maximum
        max_ind = np.argmax(mcurrent_primitive[abs_min_ind + 1:]) + (abs_min_ind - 1)

        # getting the minimum before
        # min_ind_before = np.argmin(mcurrent_primitive[:max_ind])

        # getting the minimum index after maximum
        # Making sure to include the max after so the index for the whole array
        min_ind_after = np.argmin(mcurrent_primitive[max_ind + 1:]) + (max_ind - 1)

        half_width_height = (mcurrent_primitive[max_ind] - mcurrent_primitive[abs_min_ind]) / 2
        # largest_index = np.argmax(arr[np.logical_and(arr > 2, arr < 8)])
        # Getting the closest indicies on either side of the max closest to half width
        half_width_start = np.argmax(mcurrent_primitive[::max_ind] > half_width_height)
        half_width_end = np.argmax(mcurrent_primitive[:max_ind] > half_width_height)

        # area_above_half = [i for i in range(len(mcurrent_primitive)) if mcurrent_primitive[i] > half_width_height]
        # half_width_start = area_above_half[0]
        # half_width_end = area_above_half[-1]

        # Adding start and stop coordinates appropriate to array
        half_width_height_array = np.append(half_width_height_array, [half_width_height])
        # fwhl_height = fwhl_start_stop_list.reshape((len(fwhl_start_stop_list) // 2), 2)
        fwhl_start_stop = np.append(fwhl_start_stop, [[half_width_start, half_width_end]])
        fwhl_start_stop = fwhl_start_stop.reshape((len(fwhl_start_stop) // 2), 2)

        # Determing length for primitive and appending
        full_width_length = half_width_end - half_width_start
        fwhl = np.append(fwhl, [full_width_length])

        print("Start of half width line", half_width_start)
        print("End of half width line", half_width_end)

        # # print("Half width height", half_width_height)

        # print("before max min index", min_ind_before, "value", mcurrent_primitive[min_ind_before])
        print("half width height", half_width_height)
        print("max value", max_ind, "value", mcurrent_primitive[max_ind])
        print("min value", abs_min_ind, "value", mcurrent_primitive[abs_min_ind])
        print("after max min value", min_ind_after, "value", mcurrent_primitive[min_ind_after])
        print("Length", full_width_length)
        print(mcurrent_primitive[min_ind_after])
        print()


    return fwhl, fwhl_start_stop, half_width_height_array

def full_width_half_abs_min(motor_p_full, synergy_selection):
    """Full width half maxiumum calculation
    @param: motor_p_full_full: full length numpy array of selected motor
    primitives

    @return: mean_fwhm: Mean value for the width of the primitives
    """

    number_cycles = len(motor_p_full) // 200

    # Save
    fwhl = np.array([])
    half_width_height_array = np.array([])
    fwhl_start_stop = np.empty((number_cycles, 0))

    for i in range(number_cycles):
        current_primitive = motor_p_full[i * 200: (i + 1) * 200, synergy_selection - 2]

        primitive_mask = current_primitive > 0.0

        # applying mask to exclude values which were subject to rounding errors
        mcurrent_primitive = np.asarray(current_primitive[primitive_mask])

        # getting maximum
        max_ind = np.argmax(mcurrent_primitive)

        # abs_min_ind = np.argmin(mcurrent_primitive)
        # getting the minimum before
        min_ind_before = np.argmin(mcurrent_primitive[:max_ind])

        # getting the minimum index after maximum
        # Making sure to include the max after so the index for the whole array
        min_ind_after = np.argmin(mcurrent_primitive[max_ind + 1:]) + (max_ind - 1)

        # Determing the smaller minimum to use
        if mcurrent_primitive[min_ind_before] < mcurrent_primitive[min_ind_after]:
            # print("First minimum used!")

            # Half Width formula
            half_width_height = (mcurrent_primitive[max_ind] - mcurrent_primitive[min_ind_before]) / 2

            half_width_start = np.argmax(mcurrent_primitive[::max_ind] < half_width_height) + min_ind_before
            half_width_end = np.argmax(mcurrent_primitive[:max_ind] > half_width_height)
        else:
            # print("Second minimum used")
            half_width_height = (mcurrent_primitive[max_ind] - mcurrent_primitive[min_ind_after]) / 2

        # largest_index = np.argmax(arr[np.logical_and(arr > 2, arr < 8)])
        # Getting the closest indicies on either side of the max closest to half width
        # half_width_start = np.argmax(mcurrent_primitive[::max_ind] > half_width_height)
        # half_width_end = np.argmax(mcurrent_primitive[:max_ind] > half_width_height)
        # half_width_height = (max_ind - abs_min_ind) / 2

        area_above_half = [i for i in range(len(mcurrent_primitive)) if mcurrent_primitive[i] > half_width_height]
        half_width_start = area_above_half[0]
        half_width_end = area_above_half[-1]

        # Adding start and stop coordinates appropriate to array
        half_width_height_array = np.append(half_width_height_array, [half_width_height])
        # fwhl_height = fwhl_start_stop_list.reshape((len(fwhl_start_stop_list) // 2), 2)
        fwhl_start_stop = np.append(fwhl_start_stop, [[half_width_start, half_width_end]])
        fwhl_start_stop = fwhl_start_stop.reshape((len(fwhl_start_stop) // 2), 2)

        # Determing length for primitive and appending
        full_width_length = half_width_end - half_width_start
        fwhl = np.append(fwhl, [full_width_length])

        # print("Start of half width line", half_width_start)
        # print("End of half width line", half_width_end)

        # # # print("Half width height", half_width_height)

        # print("before max min index", min_ind_before, "value", mcurrent_primitive[min_ind_before])
        # print("half width height", half_width_height)
        # print("max value", max_ind, "value", mcurrent_primitive[max_ind])
        # print("after max min value", min_ind_after, "value", mcurrent_primitive[min_ind_after])
        # print("Length", full_width_length)
        # print(mcurrent_primitive[min_ind_after])
        # print()


    return fwhl, fwhl_start_stop, half_width_height_array

def full_width_half_abs_min_scipy(motor_p_full, synergy_selection):
    """Full width half maxiumum calculation
    @param: motor_p_full_full: full length numpy array of selected motor
    primitives

    @return: mean_fwhm: Mean value for the width of the primitives
    """

    # Save
    fwhl = []
    samples = np.arange(0, len(motor_p_full))
    samples_binned = np.arange(200)
    number_cycles = len(motor_p_full) // 200
    # half_width_height_array = np.array([])
    # fwhl_start_stop = np.empty((number_cycles, 0))

    for i in range(number_cycles):
        current_primitive = motor_p_full[i * 200: (i + 1) * 200, synergy_selection - 1]

        primitive_mask = current_primitive > 0.0

        # # applying mask to exclude values which were subject to rounding errors
        mcurrent_primitive = np.asarray(current_primitive[primitive_mask])

        # Find peaks
        peaks, properties = signal.find_peaks(current_primitive, distance=40, width=2)
        max_ind = np.argmax(peaks)
        # min_ind = np.argmin(mcurrent_primitive[0:max_ind])

        # half_width_height = (mcurrent_primitive[max_ind] - mcurrent_primitive[min_ind]) / 2

        # print("Manually Calculated", half_width_height)
        max_width = properties['widths'][max_ind]
        fwhl.append(max_width)
        # fwhl_start = properties["left_ips"][max_ind]
        # fwhl_stop = properties["right_ips"][max_ind]
        # half_width_height = properties["width_heights"][max_ind]

        print("Scipy calculated", properties['widths'][max_ind])
        # print(peaks[max_ind])
    fwhl = np.asarray(fwhl)

    return fwhl

# Plotting Section
def sel_primitive_trace(motor_primitives, synergy_selection, selected_primitive_title="Output"):
    """This will plot the selected motor primitives
    @param data_input: path to csv data file
    @param synergy_selection: how many synergies you want

    @return null
    """

    # motor_primitives, motor_modules = synergy_extraction(data_input, synergy_selection)

    # Smoothen the data

    # fwhl, fwhl_start_stop, fwhl_height = full_width_half_abs_min_scipy(motor_primitives, synergy_selection)
    fwhl = []

    fwhm, half_values = fwhm(motor_primitives, synergy_selection)

    # applying mask to exclude values which were subject to rounding errors
    mcurrent_primitive = np.asarray(current_primitive[primitive_mask])

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

        # fwhl_line_start = fwhl_start_stop[i, 0]
        # fwhl_line_stop = fwhl_start_stop[i, 1]
        # plt.hlines(fwhl_height[i], fwhl_line_start, fwhl_line_stop, color='black', alpha=0.2)
        # Accumulate the trace values
        current_primitive = motor_primitives[i * 200: (i + 1) * 200, synergy_selection - 1]

        # applying mask to exclude values which were subject to rounding errors
        mcurrent_primitive = np.asarray(current_primitive[primitive_mask])

        plt.plot(samples[samples_binned], current_primitive, color='black', alpha=0.2)
        peaks, properties = signal.find_peaks(current_primitive, distance=40, width=10)
        max_ind = np.argmax(peaks)
        # print(properties['widths'][max_ind])
        max_width = properties['widths'][max_ind]
        fwhl.append(max_width)

        plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"],
            xmax=properties["right_ips"], color='black', alpha=0.2)
        primitive_trace += time_point_average

    # Calculate the average by dividing the accumulated values by the number of bins
    primitive_trace /= number_cycles



    peaks, properties = signal.find_peaks(primitive_trace, distance=40, width=15)
    # max_ind = np.argmax(peaks)

    plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"], xmax=properties["right_ips"], color = "C1")

    plt.plot(samples[samples_binned], primitive_trace, color='blue')

    # Plotting individual primitives in the background
    # selected_primitive = motor_primitives[:, synergy_selection - 2]

    # Using the order F so the values are in column order
    # binned_primitives_raw = selected_primitive.reshape((200, -1), order='F')
    # binned_primitives = ndimage.median_filter(binned_primitives_raw, size=3)
    # plt.plot(binned_primitives_raw[:, i], color='black', alpha=0.2)
    # plt.plot(binned_primitives_raw, color='black', alpha=0.2)
    # print(fwhl_start_stop[3, 1])

    # Removing axis values
    plt.xticks([])
    plt.yticks([])

    # Add a vertical line at the halfway point
    plt.axvline(x=100, color='black')

    # Adding a horizontal line for fwhl

    # fwhl_line_start = np.mean(fwhl_start_stop[:, 0])
    # fwhl_line_stop = np.mean(fwhl_start_stop[:, 1])
    # plt.hlines(y=np.mean(fwhl_height), xmin=fwhl_line_start, xmax=fwhl_line_stop, color='red')

    # Add labels for swing and stance
    # plt.text(50, -0.2 * np.max(primitive_trace), 'Swing', ha='center', va='center')
    # plt.text(150, -0.2 * np.max(primitive_trace), 'Stance', ha='center', va='center')

    # Removing top and right spines of the plot
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    plt.title(selected_primitive_title, fontsize=16, fontweight='bold')
    # plt.savefig(selected_primitive_title, dpi=300)
    # plt.show()

    fwhl = np.asarray(fwhl)
    return fwhl


def main():
    # For Turgay comparison
    synergy_selection = 1
    motor_p_data_non = pd.read_csv('./turgay_test_primitives.csv', header=None)
    motor_p_preDTX_non = motor_p_data_non.to_numpy()

    fwhl_non_syn1 = full_width_half_abs_min_scipy(motor_p_preDTX_non, synergy_selection)
    np.savetxt('./turgay_prenon1_widths.csv', fwhl_non_syn1, delimiter=',')

    synergy_selection = 2
    motor_p_data_non = pd.read_csv('./turgay_test_primitives.csv', header=None)
    motor_p_preDTX_non = motor_p_data_non.to_numpy()

    fwhl_non_syn2 = full_width_half_abs_min_scipy(motor_p_preDTX_non, synergy_selection)
    np.savetxt('./turgay_prenon2_widths.csv', fwhl_non_syn2, delimiter=',')

    synergy_selection = 3
    motor_p_data_non = pd.read_csv('./turgay_test_primitives.csv', header=None)
    motor_p_preDTX_non = motor_p_data_non.to_numpy()

    fwhl_non_syn3 = full_width_half_abs_min_scipy(motor_p_preDTX_non, synergy_selection)
    np.savetxt('./turgay_prenon3_widths.csv', fwhl_non_syn3, delimiter=',')


    # For preDTX primitives
    synergy_selection = 1
    motor_p_data_non = pd.read_csv('./predtx-non-primitives.txt', header=None)
    motor_p_preDTX_non = motor_p_data_non.to_numpy()

    motor_p_data_per = pd.read_csv('./predtx-per-primitives.txt', header=None)
    motor_p_preDTX_per = motor_p_data_per.to_numpy()

    fwhl_non_syn1 = sel_primitive_trace(motor_p_preDTX_non, synergy_selection, "M5 PreDTX without Perturbation 0.100 m/s Synergy {}".format(synergy_selection))
    np.savetxt('./prenon1_widths.csv', fwhl_non_syn1, delimiter=',')

    fwhl_per_syn1 = sel_primitive_trace(motor_p_preDTX_per, synergy_selection, "M5 PreDTX with Perturbation 0.100 m/s Synergy {}".format(synergy_selection))
    np.savetxt('./preper1_widths.csv', fwhl_per_syn1, delimiter=',')

    synergy_selection = 2
    motor_p_data_non = pd.read_csv('./predtx-non-primitives.txt', header=None)
    motor_p_preDTX_non = motor_p_data_non.to_numpy()

    motor_p_data_per = pd.read_csv('./predtx-per-primitives.txt', header=None)
    motor_p_preDTX_per = motor_p_data_per.to_numpy()

    fwhl_non_syn2 = sel_primitive_trace(motor_p_preDTX_non, synergy_selection, "M5 PreDTX without Perturbation 0.100 m/s Synergy {}".format(synergy_selection))
    np.savetxt('./prenon2_widths.csv', fwhl_non_syn2, delimiter=',')

    fwhl_per_syn2 = sel_primitive_trace(motor_p_preDTX_per, synergy_selection, "M5 PreDTX with Perturbation 0.100 m/s Synergy {}".format(synergy_selection))
    np.savetxt('./preper2_widths.csv', fwhl_per_syn2, delimiter=',')

    synergy_selection = 3
    motor_p_data_non = pd.read_csv('./predtx-non-primitives.txt', header=None)
    motor_p_preDTX_non = motor_p_data_non.to_numpy()

    motor_p_data_per = pd.read_csv('./predtx-per-primitives.txt', header=None)
    motor_p_preDTX_per = motor_p_data_per.to_numpy()

    fwhl_non_syn3 = sel_primitive_trace(motor_p_preDTX_non, synergy_selection, "M5 PreDTX without Perturbation 0.100 m/s Synergy {}".format(synergy_selection))
    np.savetxt('./prenon3_widths.csv', fwhl_non_syn3, delimiter=',')

    fwhl_per_syn3 = sel_primitive_trace(motor_p_preDTX_per, synergy_selection, "M5 PreDTX with Perturbation 0.100 m/s Synergy {}".format(synergy_selection))
    np.savetxt('./preper3_widths.csv', fwhl_per_syn3, delimiter=',')

    # For PostDTX Conditions
    synergy_selection = 1
    motor_p_data_non_post = pd.read_csv('./postdtx-non-primitives.txt', header=None)
    motor_p_preDTX_non_post = motor_p_data_non_post.to_numpy()

    motor_p_data_per_post = pd.read_csv('./postdtx-per-primitives.txt', header=None)
    motor_p_preDTX_per_post = motor_p_data_per_post.to_numpy()

    # fwhl_non_post, fwhl_non_post_start_stop, fwhl_height_non_post = full_width_half_abs_min(motor_p_preDTX_non_post, synergy_selection)
    fwhl_post_non_syn1 = sel_primitive_trace(motor_p_preDTX_non_post, synergy_selection, "M5 PostDTX without Perturbation 0.100 m/s Synergy {}".format(synergy_selection))

    # fwhl_per_post, fwhl_per_post_start_stop, fwhl_height_per_post = full_width_half_abs_min(motor_p_preDTX_per_post, synergy_selection)
    fwhl_post_per_syn1 = sel_primitive_trace(motor_p_preDTX_per_post, synergy_selection, "M5 PostDTX with Perturbation 0.100 m/s Synergy {}".format(synergy_selection))

    synergy_selection = 2
    motor_p_data_non_post = pd.read_csv('./postdtx-non-primitives.txt', header=None)
    motor_p_preDTX_non_post = motor_p_data_non_post.to_numpy()

    motor_p_data_per_post = pd.read_csv('./postdtx-per-primitives.txt', header=None)
    motor_p_preDTX_per_post = motor_p_data_per_post.to_numpy()

    # fwhl_non_post, fwhl_non_post_start_stop, fwhl_height_non_post = full_width_half_abs_min(motor_p_preDTX_non_post, synergy_selection)
    fwhl_post_non_syn2 = sel_primitive_trace(motor_p_preDTX_non_post, synergy_selection, "M5 PostDTX without Perturbation 0.100 m/s Synergy {}".format(synergy_selection))

    # fwhl_per_post, fwhl_per_post_start_stop, fwhl_height_per_post = full_width_half_abs_min(motor_p_preDTX_per_post, synergy_selection)
    fwhl_post_per_syn2 = sel_primitive_trace(motor_p_preDTX_per_post, synergy_selection, "M5 PostDTX with Perturbation 0.100 m/s Synergy {}".format(synergy_selection))

    synergy_selection = 3
    motor_p_data_non_post = pd.read_csv('./postdtx-non-primitives.txt', header=None)
    motor_p_preDTX_non_post = motor_p_data_non_post.to_numpy()

    motor_p_data_per_post = pd.read_csv('./postdtx-per-primitives.txt', header=None)
    motor_p_preDTX_per_post = motor_p_data_per_post.to_numpy()

    # fwhl_non_post, fwhl_non_post_start_stop, fwhl_height_non_post = full_width_half_abs_min(motor_p_preDTX_non_post, synergy_selection)
    fwhl_post_non_syn3 = sel_primitive_trace(motor_p_preDTX_non_post, synergy_selection, "M5 PostDTX without Perturbation 0.100 m/s Synergy {}".format(synergy_selection))

    # fwhl_per_post, fwhl_per_post_start_stop, fwhl_height_per_post = full_width_half_abs_min(motor_p_preDTX_per_post, synergy_selection)
    fwhl_post_per_syn3 = sel_primitive_trace(motor_p_preDTX_per_post, synergy_selection, "M5 PostDTX with Perturbation 0.100 m/s Synergy {}".format(synergy_selection))

    # Analysis of fwhl_lenghts

    # PreDTX Results
    predtx_results = dict()
    predtx_results.update({'PreDTX Non Syn 1': [np.mean(fwhl_non_syn1), np.std(fwhl_non_syn1)]})
    predtx_results.update({'PreDTX Non Syn 2': [np.mean(fwhl_non_syn2), np.std(fwhl_non_syn2)]})
    predtx_results.update({'PreDTX Non Syn 3': [np.mean(fwhl_non_syn3), np.std(fwhl_non_syn3)]})
    predtx_results.update({'PreDTX Per Syn 1': [np.mean(fwhl_per_syn1), np.std(fwhl_per_syn1)]})
    predtx_results.update({'PreDTX Per Syn 2': [np.mean(fwhl_per_syn2), np.std(fwhl_per_syn2)]})
    predtx_results.update({'PreDTX Per Syn 3': [np.mean(fwhl_per_syn3), np.std(fwhl_per_syn3)]})

    # sns.set()
    mean_fwhl_predtx = [value[0] for value in predtx_results.values()]
    std_fwhl_predtx = [value[1] for value in predtx_results.values()]

    # PostDTX Results
    postdtx_results = dict()
    postdtx_results.update({'PostDTX Non Syn 1': [np.mean(fwhl_post_non_syn1), np.std(fwhl_post_non_syn1)]})
    postdtx_results.update({'PostDTX Non Syn 2': [np.mean(fwhl_post_non_syn2), np.std(fwhl_post_non_syn2)]})
    postdtx_results.update({'PostDTX Non Syn 3': [np.mean(fwhl_post_non_syn3), np.std(fwhl_post_non_syn3)]})
    postdtx_results.update({'PostDTX Per Syn 1': [np.mean(fwhl_post_per_syn1), np.std(fwhl_post_per_syn1)]})
    postdtx_results.update({'PostDTX Per Syn 2': [np.mean(fwhl_post_per_syn2), np.std(fwhl_post_per_syn2)]})
    postdtx_results.update({'PostDTX Per Syn 3': [np.mean(fwhl_post_per_syn3), np.std(fwhl_post_per_syn3)]})

    mean_fwhl_postdtx = [value[0] for value in postdtx_results.values()]
    std_fwhl_postdtx = [value[1] for value in postdtx_results.values()]

    # Synergy 3 Cross comparison
    results = dict()
    results.update({'PreDTX Non Syn 3': [np.mean(fwhl_non_syn3), np.std(fwhl_non_syn3)]})
    results.update({'PreDTX Per Syn 3': [np.mean(fwhl_per_syn3), np.std(fwhl_per_syn3)]})
    results.update({'PostDTX Non Syn 3': [np.mean(fwhl_post_non_syn3), np.std(fwhl_post_non_syn3)]})
    results.update({'PostDTX Per Syn 3': [np.mean(fwhl_post_per_syn3), np.std(fwhl_post_per_syn3)]})

    mean_fwhl = [value[0] for value in results.values()]
    std_fwhl = [value[1] for value in results.values()]

    # Saving results to csv
    cycle_results_csv = 'predtx_widths.csv'

    with open(cycle_results_csv, 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the header row (optional)
        writer.writerow(['Data Point', 'Width Values'])

        # Write data from the dictionary
        for key, (mean, std_dev) in results.items():
            writer.writerow([key, mean, std_dev])

    print(f'Data has been saved to {cycle_results_csv}')
    # Plotting
    sns.set()
    predtx_trials = list(predtx_results.keys())
    postdtx_trials = list(postdtx_results.keys())
    trials = list(results.keys())
    fig, ax = plt.subplots(1, 1)
    plt.title("Full Width Half Length PreDTX for each Synergy")
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
    ax = sns.barplot(x=predtx_trials, y=mean_fwhl_predtx)
    ax.errorbar(x=predtx_trials, y=mean_fwhl_predtx, yerr=std_fwhl_predtx, capsize=2, fmt="none", c="k")
    plt.tight_layout()
    plt.show()
    plt.title("Full Width Half Length PostDTX for each Synergy")
    ax = sns.barplot(x=postdtx_trials, y=mean_fwhl_postdtx)
    ax.errorbar(x=postdtx_trials, y=mean_fwhl_postdtx, yerr=std_fwhl_postdtx, capsize=2, fmt="none", c="k")
    plt.tight_layout()
    # plt.bar(predtx_trials, mean_fwhl_predtx, yerr=std_fwhl_predtx, capsize=3)
    plt.xticks(range(len(predtx_trials)), predtx_trials, size='small')
    plt.show()
    plt.title("Full Width Half Length for Synergy 3")
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
    ax = sns.barplot(x=trials, y=mean_fwhl)
    ax.errorbar(x=trials, y=mean_fwhl, yerr=std_fwhl, capsize=2, fmt="none", c="k")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
