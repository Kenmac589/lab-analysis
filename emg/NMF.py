import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF

testfile = pd.read_excel('test file.xlsx')
A = np.array(testfile)

W2, H2 = NMF(n_components=2, init='random', random_state=0, max_iter=50).fit_transform(A)
print(W2)
print(H2)
W3, H3 = NMF(n_components=3, init='random', random_state=0, max_iter=50).fit_transform(A)
W4, H4 = NMF(n_components=4, init='random', random_state=0, max_iter=50).fit_transform(A)
W5, H5 = NMF(n_components=5, init='random', random_state=0, max_iter=50).fit_transform(A)
W6, H6 = NMF(n_components=6, init='random', random_state=0, max_iter=50).fit_transform(A)
W7, H7 = NMF(n_components=7, init='random', random_state=0, max_iter=50).fit_transform(A)

C2 = np.dot(W2, H2)
C3 = np.dot(W3, H3)
C4 = np.dot(W4, H4)
C5 = np.dot(W5, H5)
C6 = np.dot(W6, H6)
C7 = np.dot(W7, H7)

R2All = np.zeros(6)
R2All[0] = np.corrcoef(C2.flatten(), A.flatten())[0,1]**2
R2All[1] = np.corrcoef(C3.flatten(), A.flatten())[0,1]**2
R2All[2] = np.corrcoef(C4.flatten(), A.flatten())[0,1]**2
R2All[3] = np.corrcoef(C5.flatten(), A.flatten())[0,1]**2
R2All[4] = np.corrcoef(C6.flatten(), A.flatten())[0,1]**2
R2All[5] = np.corrcoef(C7.flatten(), A.flatten())[0,1]**2

X = np.array([2, 3, 4, 5, 6, 7])
plt.plot(X, R2All)
plt.show()
