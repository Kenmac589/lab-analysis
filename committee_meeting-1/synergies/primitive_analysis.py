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
import scipy as sp
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

def synergy_extraction(data_input, synergy_selection):
    """Synergy Extraction from factorized matricies
    @param data_input: path to csv data file
    @param synergy_selection:

    @return W: motor primitives
    @return H: motor modules
    @return C: factorized matrix
    """

    # Load Data
    data = pd.read_csv(data_input, header=None)
    A = data.to_numpy()

    # Define some variables about the data
    number_cycles = len(A) // 200

    # Choosing best number of components
    chosen_synergies = synergy_selection
    W, H, C = nnmf_factorize(A, chosen_synergies)

    samples = np.arange(0, len(C))
    samples_binned = np.arange(200)

    motor_modules = H
    motor_primitives = W

    return motor_primitives, motor_modules

def full_width_half_abs_min(motor_p_full, synergy_selection):
    """Full width half maxiumum calculation
    @param: motor_p_full_full: full length numpy array of selected motor primitives

    @return: mean_fwhm: Mean value for the width of the primitives
    """

    number_cycles = len(motor_p_full) // 200

    # Getting overview of all motor primitives
    plt.plot(motor_p_full[:, synergy_selection - 1])
    # plt.show()

    # Save
    full_width_length_arr = np.array([])


    for i in range(number_cycles):
        current_primitive = motor_p_full[i * 200: (i + 1) * 200, synergy_selection - 2]

        primitive_mask = current_primitive > 0.0

        # applying mask to exclude values which were subject to rounding errors
        mcurrent_primitive = np.asarray(current_primitive[primitive_mask])

        # getting maximum
        max_ind = np.argmax(mcurrent_primitive)

        # getting the minimum before
        min_ind_before = np.argmin(mcurrent_primitive[:max_ind])

        # getting the minimum index after maximum
        # Making sure to include the max after so the index for the whole array
        min_ind_after = np.argmin(mcurrent_primitive[max_ind + 1:]) + (max_ind - 1)

        # Determing the smaller minimum to use
        if mcurrent_primitive[min_ind_before] < mcurrent_primitive[min_ind_after]:
            print("First minimum used!")

            # Half Width formula
            half_width_height = (mcurrent_primitive[max_ind] - mcurrent_primitive[min_ind_before]) / 2

        else:
            print("Second minimum used")
            half_width_height = (mcurrent_primitive[max_ind] - mcurrent_primitive[min_ind_after]) / 2

        # Getting the closest indicies on either side of the max closest to half width
        half_width_start = np.argmax(mcurrent_primitive[::max_ind] > half_width_height)
        half_width_end = np.argmax(mcurrent_primitive[:max_ind] > half_width_height)

        # Determing length for primitive and appending
        full_width_length = half_width_end - half_width_start
        full_width_length_arr = np.append(full_width_length_arr, [full_width_length])

        print("Start", half_width_start)
        print("End", half_width_end)

        print("Half width height", half_width_height)

        print("before max min index", min_ind_before, "value", mcurrent_primitive[min_ind_before])
        print("max value", max_ind, "value", mcurrent_primitive[max_ind])
        print("after max min value", min_ind_after, "value", mcurrent_primitive[min_ind_after])
        print("Length", len(mcurrent_primitive))
        print(mcurrent_primitive[min_ind_after])

        # np.savetxt('primitive-{:04}.csv'.format(i), mcurrent_primitive)

        # Getting overview of all motor primitives
        plt.plot(mcurrent_primitive)

    print("Width Values", full_width_length_arr)

    mean_fwhm = np.mean(full_width_length_arr)
    return mean_fwhm

# Main Function

def main():

    directory_path = ''

    data_selection, syn_selection = './norm-emg-preDTX-100.csv', 3
    motor_p, motor_m = synergy_extraction(data_selection, syn_selection)
    average_fwhl = full_width_half_abs_min(motor_p, syn_selection)

    print(average_fwhl)
    # print(full_width_half_min)
    # print('Motor Primitives', motor_p)
    # print('Motor Modules', motor_m)
    # Calculate the number of 200-value bins

    # plt.show()

if __name__ == "__main__":
    main()
