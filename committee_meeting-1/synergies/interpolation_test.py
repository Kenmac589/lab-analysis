import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, ndimage, interpolate
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

def primitive_interpolate(motor_primitives, synergy_selection):
    samples = len(motor_primitives)
    # Showing Before
    plt.plot(motor_primitives[:, synergy_selection - 1], label='Orginal trace')
    current_primitive = motor_primitives[:, synergy_selection - 1]
    # plt.show()
    interp_func = interpolate.CubicSpline(samples, current_primitive, bc_type='natural')
    # interp_func_smooth = interpolate.make_smoothing_spline(samples, motor_primitives[:, synergy_selection - 1])
    # uniform_smooth = ndimage.uniform_filter1d(motor_primitives, size=2600)
    # motor_p_med = ndimage.median_filter(motor_primitives, size=10)
    x_new = np.linspace(0, len(motor_primitives[:, synergy_selection - 1]), 100)
    motor_p_interp = interp_func(x_new)
    # y_new = interp_func(x_new)
    plt.plot(x_new, motor_p_interp, label='Spline')
    # plt.plot(x_new[0:400], interp_func_smooth(motor_primitives[0:400, synergy_selection - 1]), label='Spline Smooth')
    # ax2.plot(x_new, uniform_smooth[:, synergy_selection - 1], label='Uniform Smooth')
    # ax3.plot(x_new, motor_p_med[:, synergy_selection - 1], label='Median Filtered')
    plt.legend(loc='best')
    plt.show()




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

# Main Function
def main():
    data_input_non, synergy_selection = './norm-emg-preDTX-100.csv', 3
    motor_p_non, motor_m_non = synergy_extraction(data_input_non, synergy_selection)
    primitive_interpolate(motor_p_non, synergy_selection)


if __name__ == "__main__":
    main()
