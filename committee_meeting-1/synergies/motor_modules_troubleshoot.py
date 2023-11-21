"""Non-Negative Matrix Factorization for Muscle Synergy Extraction
This program performs Non-Negative Matrix Factorization for determing
the appropriate number of components/muscle channels to use for muscle
synergy extraction.

"""
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

# Load Data
data = pd.read_csv("./full_width_test/norm-emg-preDTX-per.csv", header=None)
A = data.to_numpy()

# Setting various paramaters through the script I often change
selected_primitive_filename = './full_width_test/preDTX-100-per-cleaned.png'
selected_primitive_title = 'Motor Primitive for DTR-M5 preDTX with pertubration at 0.100 cleaned'
modules_and_primitives_filename = './full_width_test/preDTX-100-per-prim.png'
modules_and_primitives_title = 'Muscle Synergies for DTR-M5 preDTX with perturbation 0.100 m/s cleaned'
chosen_synergies = 7

# Define some variables about the data
number_cycles = len(A) // 200

# Choosing best number of components
W, H, C = nnmf_factorize(A, chosen_synergies)

# np.savetxt('./DTR-M5/primitives-preDTX-per-100-cleaned.csv', W, delimiter=',')
# np.savetxt('./DTR-M5/modules-preDTX-per-100-cleaned.csv', H, delimiter=',')
# np.savetxt('./DTR-M5/C3-preDTX-per-100-cleaned.csv', C, delimiter=',')

samples = np.arange(0, len(C))
samples_binned = np.arange(200)
# print("Length of A", len(A))
# print("Length of C", len(C))
# print("Length of W", len(W))

# Plot
motor_modules = H
motor_primitives = W
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
    time_point_average = motor_primitives[i * 200: (i + 1) * 200, chosen_synergies - 1]

    # Accumulate the trace values
    primitive_trace += time_point_average

# Showing individual primitives
for i in range(0, len(motor_primitives), 200):
    plt.plot(samples[samples_binned], motor_primitives[i:i + 200, chosen_synergies - 1], color='black')
    plt.title("Motor Primitives-010-per{:04}".format(i))
    plt.show()
    # plt.savefig("motor_primitives-cumulative-010-{:04}.png".format(i), dpi=300)

# Calculate the average by dividing the accumulated values by the number of bins
primitive_trace /= number_cycles

# primitives_average = np.mean(primitives_reshape, axis=1)
print("Average Primitives:", primitive_trace)
print("--------------------------------")

plt.plot(samples[samples_binned], primitive_trace, color='blue')

# Showing individual primitives
for i in range(0, len(motor_primitives), 200):
    plt.plot(samples[samples_binned], motor_primitives[i:i + 200, chosen_synergies - 1], color='black', alpha=0.2)
    plt.title("Motor Primitives-010-per{:04}".format(i))
    # plt.savefig("motor_primitives-cumulative-010-{:04}.png".format(i), dpi=300)

# Plotting individual traces in the background
for i in range(0, len(motor_primitives), 200):
    plt.plot(samples[samples_binned], motor_primitives[i:i + 200, chosen_synergies - 1], color='black', alpha=0.2)
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

# =======================================
# Presenting Data as a mutliplot figure |
# =======================================

fig, axs = plt.subplots(2, chosen_synergies, figsize=(12, 8))

# Calculate the average trace for each column
number_cycles = len(motor_primitives) // 200  # Calculate the number of 200-value bins

for col in range(chosen_synergies):
    primitive_trace = np.zeros(200)  # Initialize an array for accumulating the trace values

    # Iterate over the binned data by the number of cycles
    for i in range(number_cycles):
        # Get the data for the current bin in the current column
        time_point_average = motor_primitives[i * 200: (i + 1) * 200, col]

        # Accumulate the trace values
        primitive_trace += time_point_average

    # Calculate the average by dividing the accumulated values by the number of bins
    primitive_trace /= number_cycles

    # Plot the average trace in the corresponding subplot
    axs[0, col].plot(samples[samples_binned], primitive_trace, color='red', label='Average Trace')
    axs[0, col].set_title('Synergy {}'.format(col+1))

    # Iterate over the bins again to plot the individual bin data
    for i in range(number_cycles):
        # Get the data for the current bin in the current 0, column
        time_point_average = motor_primitives[i * 200: (i + 1) * 200, col]

        # Plot the bin data
        axs[0, col].plot(samples[samples_binned], time_point_average, label='Bin {}'.format(i+1), color='black', alpha=0.1)

    # Add vertical lines at the halfway point in each subplot
    axs[0, col].axvline(x=100, color='black')

    # Begin Presenting Motor Modules

    # Get the data for the current column
    motor_module_column_data = motor_modules[col, :]  # Select all rows for the current column

    # Set the x-axis values for the bar graph
    x_values = np.arange(len(motor_module_column_data))

    # Plot the bar graph for the current column in the corresponding subplot
    axs[1, col].bar(x_values, motor_module_column_data)

    # Remove top and right spines of each subplot
    axs[0, col].spines['top'].set_visible(False)
    axs[0, col].spines['right'].set_visible(False)
    axs[1, col].spines['top'].set_visible(False)
    axs[1, col].spines['right'].set_visible(False)

    # Remove labels on x and y axes
    axs[1, col].set_xticklabels([])
    axs[0, col].set_yticklabels([])
 
    # Remove x and y axis labels and ticks from the avg_trace subplot
    axs[0, col].set_xticks([])
    axs[0, col].set_yticks([])
    axs[0, col].set_xlabel('')
    axs[0, col].set_ylabel('')

    # Remove x and y axis labels and ticks from the motor module subplot
    axs[1, col].set_xticks(x_values, ['GM', 'Ip', 'BF', 'VL', 'St', 'TA', 'Gs', 'Gr'])
    # axs[1, col].set_xticks([])
    axs[1, col].set_yticks([])
    # axs[1, col].set_xlabel('')
    # axs[1, col].set_ylabel('')

# Adjust spacing between subplots
plt.tight_layout()
fig.suptitle(modules_and_primitives_title, fontsize=16, fontweight='bold')
plt.subplots_adjust(top=0.9)
plt.savefig(modules_and_primitives_filename, dpi=300)


# Show all the plots
plt.show()
