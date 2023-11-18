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

def synselection(data_input, num_components=np.array([2, 3, 4, 5, 6, 7])):

    # Defining set of components to use
    R2All = np.zeros(len(num_components))

    # Calculating R2 for each component
    for i in range(len(R2All)):
        W, H, C = nnmf_factorize(data_input, num_components[i])
        R2All[i] = np.corrcoef(C.flatten(), data_input.flatten())[0,1]**2
        print("$R^2$ =", i+2, ":", R2All[i])

    # Calculating correlation coefficient for each component
    corrcoef = np.zeros(len(num_components))
    for i in range(len(R2All)):
        corrcoef[i] = np.corrcoef(num_components[0:i+2], R2All[0:i+2])[0,1]
        print("r =", i+2, ":", corrcoef[i])

    # Plotting Both Methods for determining number of components
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(num_components, R2All)
    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.xlabel("Number of Components")
    plt.ylabel("$R^2$ of $C^x$ fit to original matrix")
    plt.title("Muscle Synergy Determinance by Percentage")
    plt.subplot(1, 2, 2)
    plt.scatter(num_components, corrcoef)
    plt.xlabel("Number of Components")
    plt.ylabel("Correlation Coefficient")
    plt.title("Muscle Synergy Determinance by Linearity")
    plt.show()

    # Plotting Both Methods overlapping
    plt.plot(num_components, R2All)
    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.xlabel("Number of Components")
    plt.ylabel("$R^2$ of $C^x$ fit to original matrix")
    plt.title("Muscle Synergy Determinance by Percentage")
    plt.scatter(num_components, corrcoef)
    plt.title("Muscle Synergy Determinance by Linearity and $R^2$")

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

    return full_width_length_arr

# Plotting Section
def sel_primitive_trace(data_input, synergy_selection):

    motor_primitives, motor_modules = synergy_extraction(data_input, synergy_selection)

    samples = np.arange(0, len(motor_primitives))
    samples_binned = np.arange(200)
    # print("Length of A", len(A))
    # print("Length of C", len(C))
    # print("Length of W", len(W))

    # Plot
    print("--------------------------------")
    print("motor_modules", motor_modules[:, 0])
    print("--------------------------------")
    print(motor_primitives[:, 0])
    print("--------------------------------")

    primitive_trace = np.zeros(200)

    # Labels for Modules for M5 of the 6 month group ***** ALWAYS check this
    # plt.xticks(x, ['GM', 'Ip', 'BF', 'VL', 'Gs', 'TA', 'St', 'Gr'])

    # Plotting Primitive Selected Synergy Count

    # Iterate over the bins
    for i in range(number_cycles):
        # Get the data for the current bin
        time_point_average = motor_primitives[i * 200: (i + 1) * 200, chosen_synergies-1]
        
            # Accumulate the trace values
        primitive_trace += time_point_average

    # Calculate the average by dividing the accumulated values by the number of bins
    primitive_trace /= number_cycles

    # primitives_average = np.mean(primitives_reshape, axis=1)
    print("Average Primitives:", primitive_trace)
    print("--------------------------------")

    plt.plot(samples[samples_binned], primitive_trace, color='blue')

    # Plotting individual traces in the background
    for i in range(0, len(motor_primitives), 200):
        plt.plot(samples[samples_binned], motor_primitives[i:i+200, chosen_synergies-1], color='black', alpha=0.2)
            # plt.title("Motor Primitives-010-{:04}".format(i))
            # plt.savefig("motor_primitives-cumulative-010-{:04}.png".format(i), dpi=300)

    # Removing axis values
    plt.xticks([])
    plt.yticks([])

    # Add a vertical line at the halfway point
    plt.axvline(x=100, color='black')

    # Add labels for swing and stance
    plt.text(50, -0.2 * np.max(primitive_trace), 'Swing', ha='center', va='center')
    plt.text(150, -0.2 * np.max(primitive_trace), 'Stance', ha='center', va='center')

    # Removing top and right spines of the plot
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    plt.title(selected_primitive_title, fontsize=16, fontweight='bold')
    plt.savefig(selected_primitive_filename, dpi=300)

    return

# Main Function

def main():

    # directory_path = './full_width_test/'
    # trial_list = read_all_csv(directory_path)


    data_selection_non, syn_selection_non = './full_width_test/norm-emg-preDTX-100.csv', 3
    motor_p_non, motor_m_non = synergy_extraction(data_selection_non, syn_selection_non)
    fwhl_non = full_width_half_abs_min(motor_p_non, syn_selection_non)


    data_selection_per, syn_selection_per = './full_width_test/norm-emg-preDTX-per.csv', 4
    # synselection()
    motor_p_per, motor_m_per = synergy_extraction(data_selection_per, syn_selection_per)
    fwhl_per = full_width_half_abs_min(motor_p_per, syn_selection_per)

    np.savetxt('./non.csv', fwhl_non, delimiter=',')
    np.savetxt('./per.csv', fwhl_per, delimiter=',')
    print(fwhl_non)
    print(fwhl_per)
    # print(full_width_half_min)
    # print('Motor Primitives', motor_p)
    # print('Motor Modules', motor_m)
    # Calculate the number of 200-value bins

    # plt.show()

if __name__ == "__main__":
    main()
