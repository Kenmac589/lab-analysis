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

# def normalize_emg(emg):
#     """Normalize EMG data
#     @param emg: EMG data
#     @return emg_norm: normalized EMG data
#     """
# 
#     emg_norm = (emg - np.mean(emg)) / np.std(emg)
#     return emg_norm

# Load Data
data = pd.read_csv("./norm-emg-smooth-010.csv", header=None)
A = data.to_numpy()

# Define some variables about the data
number_cycles = 15

# Defining set of components to use
num_components = np.array([2, 3, 4, 5, 6, 7])
R2All = np.zeros(len(num_components))

# Calculating R2 for each component
for i in range(len(R2All)):
    W, H, C = nnmf_factorize(A, num_components[i])
    R2All[i] = np.corrcoef(C.flatten(), A.flatten())[0,1]**2
    print ("R^2 =", i+2, ":", R2All[i])

# Calculating correlation coefficient for each component
corrcoef = np.zeros(len(num_components))
for i in range(len(R2All)):
    corrcoef[i] = np.corrcoef(num_components[0:i+2], R2All[0:i+2])[0,1]
    print("r =", i+2, ":", corrcoef[i])

# Choosing best number of components
chosen_synergies = 3
W, H, C = nnmf_factorize(A, chosen_synergies)

print(H)
print(H.shape)
print(W)
print(W.shape)
np.savetxt("W2.csv", W, delimiter=",")
samples = np.arange(0, len(C))
samples_binned = np.arange(0, 200)

# Plot
motor_modules = H
motor_primitives = W
print("--------------------------------")
print("motor_modules",motor_modules[:,0])
print("--------------------------------")
print(motor_primitives[:,0])
print("--------------------------------")

primitives_reshape = motor_primitives[:, chosen_synergies-2].reshape(200, number_cycles)
print(primitives_reshape[:5,:])
primitive_trace = np.zeros(200)
time_point_sum = 0
time_point_average = 0

for i in range(200):
    # time_point_sum = 0
    time_point_sum = np.sum(primitives_reshape[i, :number_cycles])
    time_point_average = time_point_sum / (number_cycles)
    primitive_trace[i] = time_point_average
    print(primitive_trace[i])
    # print(time_point_average)

# primitives_average = np.mean(primitives_reshape, axis=1)
print("Average Primitives:", primitive_trace)
print("--------------------------------")

for i in range(0, len(motor_primitives), 200):
    # ending_point = i+200
#    for j in range(0, 15):
#        trace_average = np.mean(motor_primitives[i, chosen_synergies-2], axis=1)
    plt.plot(samples[samples_binned], motor_primitives[i:i+200, chosen_synergies-2], color='black', alpha=0.2)
    # plt.title("Motor Primitives-010-{:04}".format(i))
    # plt.savefig("motor_primitives-cumulative-010-{:04}.png".format(i), dpi=300)

# for i in range(0, len(motor_modules), 200):
#     for j in range(0, 15):
#         plt.plot(samples[samples_binned], motor_modules[i:i+200, j], color='black', alpha=0.2)
# 
plt.plot(samples[samples_binned], primitive_trace, color='red')
plt.xticks([])
plt.yticks([])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)
plt.show()


