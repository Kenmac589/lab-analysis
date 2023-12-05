"""Non-Negative Matrix Factorization for Muscle Synergy Extraction
This program performs Non-Negative Matrix Factorization for determing
the appropriate number of components/muscle channels to use for muscle
synergy extraction.

Some functions could be more recursive however, they have been used in
applications such as synergy selection.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
from scipy import ndimage, signal, interpolate
from sklearn.decomposition import NMF

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
    data = pd.read_csv(data_input, header=0)
    A = data.to_numpy()

    # Choosing best number of components
    chosen_synergies = synergy_selection
    W, H, C = nnmf_factorize(A, chosen_synergies)

    motor_modules = H
    motor_primitives = W

    return motor_primitives, motor_modules

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
    half_width_heights = []

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
        half_width_heights.append(half_width_height)

        count_above = np.nonzero(mcurrent_primitive > half_width_height)

        fwhm_index.append(count_above)
        fwhm = np.append(fwhm, [len(count_above[0])])

    fwhm = np.asarray(fwhm)

    return half_width_heights

def show_modules(data_input, chosen_synergies, modules_filename="./output.png"):
    """
    Make sure you check the channel order!!

    """

    # =======================================
    # Presenting Data as a mutliplot figure |
    # =======================================
    motor_primitives, motor_modules = synergy_extraction(data_input, chosen_synergies)
    channel_order = ['GM', 'Ip', 'BF', 'VL', 'St', 'TA', 'Gs', 'Gr']

    fig, axs = plt.subplots(chosen_synergies, 1, figsize=(4, 10))

    # Calculate the average trace for each column
    samples = np.arange(0, len(motor_primitives))
    samples_binned = np.arange(200)
    number_cycles = len(motor_primitives) // 200

    for col in range(chosen_synergies):
        primitive_trace = np.zeros(200)  # Initialize an array for accumulating the trace values

        # Begin Presenting Motor Modules

        # Get the data for the current column
        motor_module_column_data = motor_modules[col, :]  # Select all rows for the current column

        # Set the x-axis values for the bar graph
        x_values = np.arange(len(motor_module_column_data))

        # Plot the bar graph for the current column in the corresponding subplot
        axs[col].bar(x_values, motor_module_column_data)

        # Remove top and right spines of each subplot

        # Remove x and y axis labels and ticks from the avg_trace subplot
        axs[col].set_xticks([])
       axs[col].set_yticks([])
        axs[col].set_xlabel('')
        axs[col].set_ylabel('')
        axs[col].spines['top'].set_visible(False)
        axs[col].spines['right'].set_visible(False)

        # Remove x and y axis labels and ticks from the motor module subplot
        axs[col].set_xticks(x_values, channel_order)
        axs[col].set_yticks([])
        # axs[1, col].set_xlabel('')
        # axs[1, col].set_ylabel('')

    # Adjust spacing between subplots
    plt.tight_layout()
    # fig.suptitle(synergies_title, fontsize=16, fontweight='bold')
    plt.savefig(modules_filename, dpi=300)
    # plt.subplots_adjust(top=0.9)
    plt.show()

def show_modules_dtr(data_input, chosen_synergies, modules_filename="./output.png"):
    """
    Make sure you check the channel order!!

    """

    # =======================================
    # Presenting Data as a mutliplot figure |
    # =======================================
    motor_primitives, motor_modules = synergy_extraction(data_input, chosen_synergies)
    channel_order_dtr = ['GM', 'Ip', 'BF', 'VL', 'Gs', 'TA', 'St', 'Gr']

    fig, axs = plt.subplots(chosen_synergies, 1, figsize=(4, 10))

    # Calculate the average trace for each column
    samples = np.arange(0, len(motor_primitives))
    samples_binned = np.arange(200)
    number_cycles = len(motor_primitives) // 200

    for col in range(chosen_synergies):
        primitive_trace = np.zeros(200)  # Initialize an array for accumulating the trace values

        # Begin Presenting Motor Modules

        # Get the data for the current column
        motor_module_column_data = motor_modules[col, :]  # Select all rows for the current column

        # Set the x-axis values for the bar graph
        x_values = np.arange(len(motor_module_column_data))

        # Plot the bar graph for the current column in the corresponding subplot
        axs[col].bar(x_values, motor_module_column_data)

        # Remove top and right spines of each subplot

        # Remove x and y axis labels and ticks from the avg_trace subplot
        axs[col].set_xticks([])
        axs[col].set_yticks([])
        axs[col].set_xlabel('')
        axs[col].set_ylabel('')
        axs[col].spines['top'].set_visible(False)
        axs[col].spines['right'].set_visible(False)

        # Remove x and y axis labels and ticks from the motor module subplot
        axs[col].set_xticks(x_values, channel_order_dtr)
        axs[col].set_yticks([])
        # axs[1, col].set_xlabel('')
        # axs[1, col].set_ylabel('')

    # Adjust spacing between subplots
    plt.tight_layout()
    # fig.suptitle(synergies_title, fontsize=16, fontweight='bold')
    plt.savefig(modules_filename, dpi=300)
    # plt.subplots_adjust(top=0.9)
    plt.show()

def sel_primitive_trace_with_fwhm(motor_primitives, synergy_selection, selected_primitive_title="Output"):
    """This will plot the selected motor primitives
    @param data_input: path to csv data file
    @param synergy_selection: how many synergies you want

    @return null
    """

    # motor_primitives, motor_modules = synergy_extraction(data_input, synergy_selection)

    # Smoothen the data

    fwhm_line = fwhm(motor_primitives, synergy_selection)

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
        plt.plot(samples[samples_binned], current_primitive, color='black', alpha=0.2)
        plt.hlines(y=fwhm_line[i], xmin=0, xmax=len(current_primitive), color='black', alpha=0.2)

        primitive_trace += time_point_average

    # Calculate the average by dividing the accumulated values by the number of bins
    primitive_trace /= number_cycles

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
    plt.text(50, -0.2 * np.max(primitive_trace), 'Swing', ha='center', va='center')
    plt.text(150, -0.2 * np.max(primitive_trace), 'Stance', ha='center', va='center')

    # Removing top and right spines of the plot
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    plt.title(selected_primitive_title, fontsize=12, fontweight='bold')
    # plt.savefig(selected_primitive_title, dpi=300)
    plt.show()

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
    # fwhm = []

    # fwhm, half_values = fwhm(motor_primitives, synergy_selection)

    # applying mask to exclude values which were subject to rounding errors
    # mcurrent_primitive = np.asarray(current_primitive[primitive_mask])

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

        primitive_mask = current_primitive > 0.0
        # applying mask to exclude values which were subject to rounding errors
        mcurrent_primitive = np.asarray(current_primitive[primitive_mask])

        plt.plot(samples[samples_binned], current_primitive, color='black', alpha=0.2)
        # peaks, properties = signal.find_peaks(current_primitive, distance=40, width=10)
        # max_ind = np.argmax(peaks)
        # print(properties['widths'][max_ind])
        # max_width = properties['widths'][max_ind]
        # fwhl.append(max_width)

        # plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"],
            # xmax=properties["right_ips"], color='black', alpha=0.2)
        primitive_trace += time_point_average

    # Calculate the average by dividing the accumulated values by the number of bins
    primitive_trace /= number_cycles

    plt.plot(samples[samples_binned], primitive_trace, color='blue')

    # Plotting individual primitives in the background
    selected_primitive = motor_primitives[:, synergy_selection - 2]

    # Using the order F so the values are in column order
    binned_primitives_raw = selected_primitive.reshape((200, -1), order='F')
    # binned_primitives = ndimage.median_filter(binned_primitives_raw, size=3)
    plt.plot(binned_primitives_raw[:, i], color='black', alpha=0.2)
    plt.plot(binned_primitives_raw, color='black', alpha=0.2)

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
    plt.show()

    # fwhl = np.asarray(fwhl)
    # return fwhl


def main():

    # For WT
    synergy_selection = 1
    motor_p_data_non = pd.read_csv('./com-non-primitives.txt', header=0)
    motor_p_wt_non = motor_p_data_non.to_numpy()

    motor_p_data_per = pd.read_csv('./com-per-primitives.txt', header=0)
    motor_p_wt_per = motor_p_data_per.to_numpy()

    fwhl_wt_non_syn1 = sel_primitive_trace_with_fwhm(motor_p_wt_non, synergy_selection, "WT without Perturbation Synergy {}".format(synergy_selection))
    # # np.savetxt('./prenon1_widths.csv', fwhl_non_syn1, delimiter=',')

    fwhl_wt_per_syn1 = sel_primitive_trace_with_fwhm(motor_p_wt_per, synergy_selection, "WT with Perturbation Synergy {}".format(synergy_selection))
    # # np.savetxt('./preper1_widths.csv', fwhl_per_syn1, delimiter=',')

    synergy_selection = 2
    motor_p_data_non = pd.read_csv('./com-non-primitives.txt', header=0)
    motor_p_wt_non = motor_p_data_non.to_numpy()

    motor_p_data_per = pd.read_csv('./com-per-primitives.txt', header=0)
    motor_p_wt_per = motor_p_data_per.to_numpy()

    fwhl_wt_non_syn2 = sel_primitive_trace_with_fwhm(motor_p_wt_non, synergy_selection, "WT without Perturbation Synergy {}".format(synergy_selection))
    # # np.savetxt('./prenon2_widths.csv', fwhl_non_syn2, delimiter=',')

    fwhl_wt_per_syn2 = sel_primitive_trace_with_fwhm(motor_p_wt_per, synergy_selection, "WT with Perturbation Synergy {}".format(synergy_selection))
    # # np.savetxt('./preper2_widths.csv', fwhl_per_syn2, delimiter=',')

    synergy_selection = 3
    motor_p_data_non = pd.read_csv('./com-non-primitives.txt', header=0)
    motor_p_wt_non = motor_p_data_non.to_numpy()

    motor_p_data_per = pd.read_csv('./com-per-primitives.txt', header=0)
    motor_p_wt_per = motor_p_data_per.to_numpy()

    fwhl_wt_non_syn3 = sel_primitive_trace_with_fwhm(motor_p_wt_non, synergy_selection, "WT without Perturbation Synergy {}".format(synergy_selection))

    fwhl_wt_per_syn3 = sel_primitive_trace_with_fwhm(motor_p_wt_per, synergy_selection, "WT with Perturbation Synergy {}".format(synergy_selection))

    # Analysis of fwhl_lenghts


    # Showing Modules
    show_modules('./norm-emg-com-non.csv', 3, "../figures/CoM-M1/mod_non.png")
    show_modules('./norm-emg-com-per.csv', 3, "../figures/CoM-M1/mod_per.png")
    show_modules_dtr('./norm-emg-preDTX-100.csv', 3, "../figures/dtr-m5/modules/mod_prenon.png")
    show_modules_dtr('./norm-emg-preDTX-per.csv', 3, "../figures/dtr-m5/modules/mod_preper.png")
    show_modules_dtr('./norm-postdtx-non.csv', 3, "../figures/dtr-m5/modules/mod_postnon.png")
    show_modules_dtr('./norm-postdtx-per.csv', 3, "../figures/dtr-m5/modules/mod_postper.png")


if __name__ == "__main__":
    main()
