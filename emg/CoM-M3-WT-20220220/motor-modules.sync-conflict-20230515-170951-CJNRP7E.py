"""Non-Negative Matrix Factorization for Muscle Synergy Extraction

This program performs Non-Negative Matrix Factorization for determing
the appropriate number of components/muscle channels to use for muscle
synergy extraction.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
data = pd.read_csv("./norm-emg-smooth-010.csv", header=None)
A = data.to_numpy()

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
W, H, C = nnmf_factorize(A, 3)

print(H)
print(H.shape)
print(W)
print(W.shape)
np.savetxt("W3.csv", W, delimiter=",")
samples = np.arange(0, len(C))
samples_binned = np.arange(0, 200)

# Plot
motor_modules = H
motor_primitives = W

# Taking every 200 values from motor_primitives and saving as new array
# fig, axs = plt.subplots(2, 5, figsize=(3, 5))
# for i, ax in enumerate(axs.flat):
#     for j in range(0, len(motor_primitives), 200):
#         ax.plot(samples[samples_binned], motor_primitives[j:j+200])

for i in range(3):
    for j in range(0, len(motor_primitives), 200):
        starting_point = j
        plt.plot(samples[samples_binned], motor_primitives[j:j+200, i])
        plt.title("Motor Primitives-010-{:04}".format(i))
        plt.savefig("motor_primitives-cumulative-010-{:04}.png".format(i), dpi=300)

print(motor_modules[:,0])
print(motor_primitives[:,0])
#plt.plot (samples, motor_primitives[:,0])
# plt.title("Motor Primitives-010")
# plt.savefig("motor_primitives-cumulative-010.png", dpi=300)
# plt.tight_layout()
# plt.show()
