"""Non-Negative Matrix Factorization for Muscle Synergy Extraction
This program performs Non-Negative Matrix Factorization for determing
the appropriate number of components/muscle channels to use for muscle
synergy extraction.

Some functions could be more recursive however, they have been used in
applications such as synergy selection.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.decomposition import NMF
from decimal import *

getcontext().prec = 28

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

def full_width_half_high_min(motor_p_full, synergy_selection):
    """Full width half maxiumum calculation
    @param: motor_p_full: full length numpy array of selected motor primitives

    @return: mean_fwhm: Mean value for the width of the primitives
    """

    number_cycles = len(motor_p_full) // 200  # Calculate the number of 200-value bins

    indivdual_primitives = np.split(motor_p_full[:, synergy_selection - 1], 200)

    # findin

    for i in range(number_cycles):
        primitive_minimum = 0
        primitive_maximum = 0

        # samples[samples_binned], motor_p_full[i:i+200, syn_selection-1]

    mean_fwhm = indivdual_primitives
    return mean_fwhm

def full_width_half_abs_min(motor_p_full, synergy_selection):
    """Full width half maxiumum calculation
    @param: motor_p_full_full: full length numpy array of selected motor primitives

    @return: mean_fwhm: Mean value for the width of the primitives
    """

    number_cycles = len(motor_p_full) // 200  # Calculate the number of 200-value bins

    # Getting overview of all motor primitives

    # np.savetxt('motor_p_full.csv', motor_p_full, delimiter=',')
    # plt.plot(motor_p_full[:, synergy_selection - 1])
    # plt.show()

    indivdual_primitives_list = np.split(motor_p_full[:, synergy_selection - 1], 200)
    indivdual_primitives = np.array(indivdual_primitives_list)
    print(indivdual_primitives)

    # Find minimum and maximum values in each primitive
    # min_vals = [np.amin(i) for i in indivdual_primitives]
    # max_vals = [np.amax(i) for i in indivdual_primitives]


    # findin
    # full_width_half = np.zeros(number_cycles)

    for i in range(number_cycles):
        current_primitive = motor_p_full[i * 200: (i + 1) * 200, synergy_selection - 2]

        primitive_mask = current_primitive > 0.0

        # applying mask to exclude values which were subject to rounding errors
        mcurrent_primitive = current_primitive[primitive_mask]

        # getting maximum
        max_ind = np.argmax(mcurrent_primitive)
        min_ind = np.argmin(mcurrent_primitive)

        # getting the minimum before
        min_ind_before = np.argmin(mcurrent_primitive[:max_ind])

        # getting the minimum after
        min_ind_after = np.argmin(mcurrent_primitive[max_ind + 1:])

        if min_ind_before < min_ind_after:
            half_width_height = (mcurrent_primitive[max_ind] - mcurrent_primitive[min_ind_before]) / 2

            half_width_start = np.argmax(mcurrent_primitive[:max_ind] > half_width_height)
            half_width_end = np.argmax(mcurrent_primitive[max_ind + 1:] < half_width_height)
            print("Start", half_width_start)
            print("End", half_width_end)


            # half_width_start = mcurrent_primitive.argwhere(mcurrent_primitive[:max_ind] > half_width_height)
            # half_width_end = mcurrent_primitive.argwhere(mcurrent_primitive[max_ind + 1:] < half_width_height)
            print(half_width_height)
        else:
            half_width_height = (mcurrent_primitive[max_ind] - mcurrent_primitive[min_ind_after]) / 2
            half_width_start = mcurrent_primitive.argwhere(mcurrent_primitive[max_ind + 1:] < half_width_height)
            half_width_end = mcurrent_primitive.argwhere(mcurrent_primitive[:max_ind] > half_width_height)

            half_width_start = np.argmax(mcurrent_primitive[:max_ind] > half_width_height)
            half_width_end = np.argmax(mcurrent_primitive[max_ind + 1:] < half_width_height)
            print("Start", half_width_start)
            print("End", half_width_end)

        print("prim max", np.max(mcurrent_primitive), "at index", max_ind)
        print("prim min", np.min(mcurrent_primitive), "at index", min_ind)

        print("before", min_ind_before)
        print("after", min_ind_after)

    mean_fwhm = indivdual_primitives
    return mean_fwhm

# Main Function

def main():

    data_selection = './norm-emg-preDTX-100.csv'
    syn_selection = 3
    motor_p, motor_m = synergy_extraction(data_selection, syn_selection)

    # print(full_width_half_mean)
    # print('Motor Primitives', motor_p)
    # print('Motor Modules', motor_m)
    # Calculate the number of 200-value bins

    number_cycles = len(motor_p) // 200

    # Getting overview of all motor primitives
    plt.plot(motor_p[:, syn_selection - 1])
    plt.show()

    # indivdual_primitives_list = np.split(motor_p[:, syn_selection - 1], 200)
    # indivdual_primitives = np.array(indivdual_primitives_list)
    # print()
    # print("Individial Primitives test output")
    # print(indivdual_primitives)
    # print()

    for i in range(number_cycles):
        current_primitive = motor_p[i * 200: (i + 1) * 200, syn_selection - 1]

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

        if mcurrent_primitive[min_ind_before] < mcurrent_primitive[min_ind_after]:
            print("First minimum used!")
            half_width_height = (mcurrent_primitive[max_ind] - mcurrent_primitive[min_ind_before]) / 2

            # Old directions
            # half_width_start = np.argmax(mcurrent_primitive[:max_ind] > half_width_height)
            # half_width_end = np.argmax(mcurrent_primitive[max_ind + 1:] < half_width_height)

            # New Test
            half_width_start = np.argmax(mcurrent_primitive[:max_ind] < half_width_height)
            half_width_end = np.argmax(mcurrent_primitive[max_ind + 1:] > half_width_height)

            print("Start", half_width_start)
            print("End", half_width_end)

            print("Half width height", half_width_height)
        else:
            print("Second minimum used")
            half_width_height = (mcurrent_primitive[max_ind] - mcurrent_primitive[min_ind_after]) / 2

            # Old Method
            half_width_start = np.argmax(mcurrent_primitive[max_ind + 1:] > half_width_height)
            half_width_end = np.argmax(mcurrent_primitive[:max_ind] > half_width_height)

            print("Start", half_width_start)
            print("End", half_width_end)

            print("Half width height", half_width_height)

        # print("prim max", np.max(mcurrent_primitive), "at index", max_ind)
        # print("prim min", np.min(mcurrent_primitive), "at index", min_ind)

        print("before max min index", min_ind_before, "value", mcurrent_primitive[min_ind_before])
        print("max value", max_ind, "value", mcurrent_primitive[max_ind])
        print("after max min value", min_ind_after, "value", mcurrent_primitive[min_ind_after])
        print("Length", len(mcurrent_primitive))
        print(mcurrent_primitive[min_ind_after])
        print(type(mcurrent_primitive))
        print()



        # Getting overview of all motor primitives
        plt.plot(mcurrent_primitive)

    plt.show()

if __name__ == "__main__":
    main()
