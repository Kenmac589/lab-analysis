import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF

normalizedData = pd.read_csv('./normalized-emg.csv', header=None)

# Spittling up data by swing and stance (every 200 samples)
chunk_indecies = np.arange(0, len(normalizedData), 200)

# Split data into chunks
normalizedData_chunks = [normalizedData.iloc[i:i+200] for i in chunk_indecies]

# print first few rows of each chunk
for i, chunk in enumerate(normalizedData_chunks):
    print(f"Chunk {i+1}:")
    print(chunk.head())
    print("\n")

# for i, chunk in enumerate(normalizedData_chunks):
#     A = normalizedData_chunks[i].to_numpy()
#     # Number of components
#     nmf = NMF(n_components=2, init='random', random_state=0)
#     W2= nmf.fit_transform(A)
#     H2 = nmf.components_
# 
#     nmf = NMF(n_components=3, init='random', random_state=0)
#     W3 = nmf.fit_transform(A)
#     H3 = nmf.components_
# 
#     nmf = NMF(n_components=4, init='random', random_state=0)
#     W4 = nmf.fit_transform(A)
#     H4 = nmf.components_
# 
#     fig, axs = plt.subplots(1, 4, figsize=(10, 4))
#     for i, ax in enumerate(axs):
#         ax.plot(H4[i])
#         ax.set_xlabel('Dimension')
#         ax.set_ylabel('Weight')
#         ax.set_title(f'Component {i+1}')
#     
#     plt.tight_layout()
#     plt.show()

# For testing
A = normalizedData_chunks[0].to_numpy()
# Number of components
nmf = NMF(n_components=7, init='random', random_state=0)
W = nmf.fit_transform(A)
H = nmf.components_

fig, axs = plt.subplots(1, 7, figsize=(20, 3))
for i, ax in enumerate(axs):
    ax.plot(H[i])
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Weight')
    ax.set_title(f'Component {i+1}')

plt.tight_layout()
plt.show()




# Plot each chunk
# for i, chunk in enumerate(normalizedData_chunks):
#     plt.plot(chunk)
#     plt.show()

