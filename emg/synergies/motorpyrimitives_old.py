"""Non-Negative Matrix Factorization for Muscle Synergy Extraction
This program performs Non-Negative Matrix Factorization for determing
the appropriate number of components/muscle channels to use for muscle
synergy extraction.

Some functions could be more recursive however, they have been used in
applications such as synergy selection.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate, ndimage, sig
from sklearn.decomposition import NMF


def nnmf_factorize(A, k):
    """Non-Negative Matrix Factorization for Muscle Synergy Extraction
    @param A: input matrix
    @param k: number of components (muscle channels)

    @return W: motor primitives
    @return H: motor modules
    @return C: factorized matrix
    """
    nmf = NMF(n_components=k, init="random", random_state=0)
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


def full_width_half_first_min(motor_p_full, synergy_selection):
    """full width half maxiumum calculation
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
        current_primitive = motor_p_full[i * 200 : (i + 1) * 200, synergy_selection - 2]

        primitive_mask = current_primitive > 0.0

        # applying mask to exclude values which were subject to rounding errors
        mcurrent_primitive = np.asarray(current_primitive[primitive_mask])

        # Dealing with local maxima issues at ends of primitives
        # diff_mcurrent = np.diff(mcurrent_primitive_full, axis=0)
        # mcurrent_primitive = mcurrent_primitive_full[np.arange(mcurrent_primitive_full.shape[0]), diff_mcurrent]

        abs_min_ind = np.argmin(mcurrent_primitive)

        # getting maximum
        max_ind = np.argmax(mcurrent_primitive[abs_min_ind + 1 :]) + (abs_min_ind - 1)

        # getting the minimum before
        # min_ind_before = np.argmin(mcurrent_primitive[:max_ind])

        # getting the minimum index after maximum
        # Making sure to include the max after so the index for the whole array
        min_ind_after = np.argmin(mcurrent_primitive[max_ind + 1 :]) + (max_ind - 1)

        half_width_height = (
            mcurrent_primitive[max_ind] - mcurrent_primitive[abs_min_ind]
        ) / 2
        # largest_index = np.argmax(arr[np.logical_and(arr > 2, arr < 8)])
        # Getting the closest indicies on either side of the max closest to half width
        half_width_start = np.argmax(mcurrent_primitive[::max_ind] > half_width_height)
        half_width_end = np.argmax(mcurrent_primitive[:max_ind] > half_width_height)

        # area_above_half = [i for i in range(len(mcurrent_primitive)) if mcurrent_primitive[i] > half_width_height]
        # half_width_start = area_above_half[0]
        # half_width_end = area_above_half[-1]

        # Adding start and stop coordinates appropriate to array
        half_width_height_array = np.append(
            half_width_height_array, [half_width_height]
        )
        # fwhl_height = fwhl_start_stop_list.reshape((len(fwhl_start_stop_list) // 2), 2)
        fwhl_start_stop = np.append(
            fwhl_start_stop, [[half_width_start, half_width_end]]
        )
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
        print(
            "after max min value",
            min_ind_after,
            "value",
            mcurrent_primitive[min_ind_after],
        )
        print("Length", full_width_length)
        print(mcurrent_primitive[min_ind_after])
        print()

    return fwhl, fwhl_start_stop, half_width_height_array


def full_width_half_abs_min(motor_p_full, synergy_selection):
    """full width half maxiumum calculation
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
        current_primitive = motor_p_full[i * 200 : (i + 1) * 200, synergy_selection - 2]

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
        min_ind_after = np.argmin(mcurrent_primitive[max_ind + 1 :]) + (max_ind - 1)

        # Determing the smaller minimum to use
        if mcurrent_primitive[min_ind_before] < mcurrent_primitive[min_ind_after]:
            # print("First minimum used!")

            # Half Width formula
            half_width_height = (
                mcurrent_primitive[max_ind] - mcurrent_primitive[min_ind_before]
            ) / 2

            half_width_start = (
                np.argmax(mcurrent_primitive[::max_ind] < half_width_height)
                + min_ind_before
            )
            half_width_end = np.argmax(mcurrent_primitive[:max_ind] > half_width_height)
        else:
            # print("Second minimum used")
            half_width_height = (
                mcurrent_primitive[max_ind] - mcurrent_primitive[min_ind_after]
            ) / 2

        # largest_index = np.argmax(arr[np.logical_and(arr > 2, arr < 8)])
        # Getting the closest indicies on either side of the max closest to half width
        # half_width_start = np.argmax(mcurrent_primitive[::max_ind] > half_width_height)
        # half_width_end = np.argmax(mcurrent_primitive[:max_ind] > half_width_height)
        # half_width_height = (max_ind - abs_min_ind) / 2

        area_above_half = [
            i
            for i in range(len(mcurrent_primitive))
            if mcurrent_primitive[i] > half_width_height
        ]
        half_width_start = area_above_half[0]
        half_width_end = area_above_half[-1]

        # Adding start and stop coordinates appropriate to array
        half_width_height_array = np.append(
            half_width_height_array, [half_width_height]
        )
        # fwhl_height = fwhl_start_stop_list.reshape((len(fwhl_start_stop_list) // 2), 2)
        fwhl_start_stop = np.append(
            fwhl_start_stop, [[half_width_start, half_width_end]]
        )
        fwhl_start_stop = fwhl_start_stop.reshape((len(fwhl_start_stop) // 2), 2)

        # Determing length for primitive and appending
        full_width_length = half_width_end - half_width_start
        fwhl = np.append(fwhl, [full_width_length])

        print("Start of half width line", half_width_start)
        print("End of half width line", half_width_end)

        # # print("Half width height", half_width_height)

        print(
            "before max min index",
            min_ind_before,
            "value",
            mcurrent_primitive[min_ind_before],
        )
        print("half width height", half_width_height)
        print("max value", max_ind, "value", mcurrent_primitive[max_ind])
        print(
            "after max min value",
            min_ind_after,
            "value",
            mcurrent_primitive[min_ind_after],
        )
        print("Length", full_width_length)
        print(mcurrent_primitive[min_ind_after])
        print()

    return fwhl, fwhl_start_stop, half_width_height_array


def full_width_half_abs_min_scipy(motor_p_full, synergy_selection):
    """full width half maxiumum calculation
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
        current_primitive = motor_p_full[i * 200 : (i + 1) * 200, synergy_selection - 1]

        # Find peaks
        peaks, properties = sig.find_peaks(current_primitive, distance=40, width=3)
        max_ind = np.argmax(peaks)
        # min_ind = np.argmin(mcurrent_primitive[0:max_ind])

        fwhl = properties["widths"][max_ind]
        fwhl_start = properties["left_ips"][max_ind]
        fwhl_stop = properties["right_ips"][max_ind]
        half_width_height = properties["width_heights"][max_ind]

        fwhl_start_stop = np.append(fwhl_start_stop, [[fwhl_start, fwhl_stop]])
        half_width_height_array = np.append(
            half_width_height_array, [half_width_height]
        )

        print("Scipy calculated", properties["widths"][max_ind])
        # print(peaks[max_ind])

    return fwhl, fwhl_start_stop, half_width_height_array


# Plotting Section
def sel_primitive_trace(
    data_input, synergy_selection, selected_primitive_title="Output"
):
    """This will plot the selected motor primitives
    @param data_input: path to csv data file
    @param synergy_selection: how many synergies you want

    @return null
    """

    motor_primitives, motor_modules = synergy_extraction(data_input, synergy_selection)

    # Smoothen the data

    fwhl, fwhl_start_stop, fwhl_height = full_width_half_abs_min(
        motor_primitives, synergy_selection
    )

    samples = np.arange(0, len(motor_primitives))
    samples_binned = np.arange(200)
    number_cycles = len(motor_primitives) // 200

    # Plot
    primitive_trace = np.zeros(200)

    # Plotting Primitive Selected Synergy Count

    # Iterate over the bins
    for i in range(number_cycles):
        # Get the data for the current bin

        time_point_average = motor_primitives[
            i * 200 : (i + 1) * 200, synergy_selection - 2
        ]

        fwhl_line_start = fwhl_start_stop[i, 0]
        fwhl_line_stop = fwhl_start_stop[i, 1]
        plt.hlines(
            fwhl_height[i], fwhl_line_start, fwhl_line_stop, color="black", alpha=0.2
        )
        # Accumulate the trace values
        primitive_trace += time_point_average

    # Calculate the average by dividing the accumulated values by the number of bins
    primitive_trace /= number_cycles

    plt.plot(samples[samples_binned], primitive_trace, color="blue")

    # Plotting individual primitives inthe background
    selected_primitive = motor_primitives[:, synergy_selection - 2]

    # Using the order F so the values are in column order
    binned_primitives_raw = selected_primitive.reshape((200, -1), order="F")

    binned_primitives = ndimage.median_filter(binned_primitives_raw, size=3)
    plt.plot(binned_primitives[:, i], color="black", alpha=0.2)
    # print(fwhl_start_stop[3, 1])

    # Removing axis values
    plt.xticks([])
    plt.yticks([])

    # Add a vertical line at the halfway point
    plt.axvline(x=100, color="black")

    # Adding a horizontal line for fwhl

    fwhl_line_start = np.mean(fwhl_start_stop[:, 0])
    fwhl_line_stop = np.mean(fwhl_start_stop[:, 1])
    plt.hlines(
        y=np.mean(fwhl_height), xmin=fwhl_line_start, xmax=fwhl_line_stop, color="red"
    )

    # Add labels for swing and stance
    plt.text(50, -0.2 * np.max(primitive_trace), "Swing", ha="center", va="center")
    plt.text(150, -0.2 * np.max(primitive_trace), "Stance", ha="center", va="center")

    # Removing top and right spines of the plot
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(True)
    plt.gca().spines["left"].set_visible(True)
    plt.title(selected_primitive_title, fontsize=16, fontweight="bold")
    # plt.savefig(selected_primitive_filename, dpi=300)
    plt.show()


# Main Function


def main():

    data_selection_non, syn_selection_non = (
        "./full_width_test/norm-emg-preDTX-100.csv",
        3,
    )
    motor_p_non, motor_m_non = synergy_extraction(data_selection_non, syn_selection_non)
    fwhl_non, fwhl_non_start_stop, fwhl_height_non = full_width_half_abs_min(
        motor_p_non, syn_selection_non
    )

    data_selection_per, syn_selection_per = (
        "./full_width_test/norm-emg-preDTX-per.csv",
        3,
    )
    motor_p_per, motor_m_per = synergy_extraction(data_selection_per, syn_selection_per)
    fwhl_per, fwhl_per_start_stop, fwhl_height_per = full_width_half_abs_min(
        motor_p_per, syn_selection_per
    )

    sel_primitive_trace(
        data_selection_non, syn_selection_non, "M5 PreDTX Non-pertubation 0.100m/s"
    )
    sel_primitive_trace(
        data_selection_per, syn_selection_per, "M5 PreDTX with Pertubation 0.100m/s"
    )

    # Post DTX Group
    data_selection_non_post, syn_selection_non_post = (
        "./full_width_test/norm-emg-postDTX-100.csv",
        2,
    )
    motor_p_non_post, motor_m_non_post = synergy_extraction(
        data_selection_non_post, syn_selection_non_post
    )
    fwhl_non_post, fwhl_non_start_stop_post, fwhl_height_non_post = (
        full_width_half_abs_min(motor_p_non_post, syn_selection_non_post)
    )

    sel_primitive_trace(
        data_selection_non_post,
        syn_selection_non_post,
        "M5 PostDTX with Pertubation 0.100m/s",
    )


if __name__ == "__main__":
    main()
