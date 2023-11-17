---
title: "Muscle Synergies"
bibliography: "/Users/kenzie_mackinnon/Documents/zotero_m1_macbook.bib"
---

## Background

- From @Rabbi2020

Muscle activation patterns (x) can be reconstructed by multiplying the muscle synergy weights (w) and synergy excitation primitives (h).

Eqn 1: $$x(t)=\mathop{\sum }\limits_{i=1}^{N}{w}_{i}{h}_{i}(t)+e(t)$$ 

where e represents reconstruction error. Eqn 1 can be writen in vector form as

Eqn 2: $${\boldsymbol{x}}={\boldsymbol{wh}}+{\boldsymbol{e}}$$


Breaking down Turgay's MATLAB script

```matlab
uiimport('-file');
A=table2array(testfile);

[W2,H2]=nnmf(A, 2, 'replicates', 50, 'algorithm', 'als');
[W3,H3]=nnmf(A, 3, 'replicates', 50, 'algorithm', 'als');
[W4,H4]=nnmf(A, 4, 'replicates', 50, 'algorithm', 'als');
[W5,H5]=nnmf(A, 5, 'replicates', 50, 'algorithm', 'als');
[W6,H6]=nnmf(A, 6, 'replicates', 50, 'algorithm', 'als');
[W7,H7]=nnmf(A, 7, 'replicates', 50, 'algorithm', 'als');

```

- The n * m matrix A is transformed into Non-negative factors.
    - W (*n*-by-k): Non-negative left factor A, is the number of rows of A, and k is the second input argument of `nnmf`
        - `W` and `H` are normalized so that the *row* of `H` have unit length. The columns of `W` are ordered by decreasing length.
    - H (k-by-*m*): Non-negative right factor of A, returned as as (k-by-*m*) matrix. `k` is the second argument of `nnmf`, and *m* is the number of columns of `A`

```matlab
C2=W2*H2;
C3=W3*H3;
C4=W4*H4;
C5=W5*H5;
C6=W6*H6;
C7=W7*H7;

R2All(1) =corr(C2(:),A(:))^2;
R2All(2) =corr(C3(:),A(:))^2;
R2All(3) =corr(C4(:),A(:))^2;
R2All(4) =corr(C5(:),A(:))^2;
R2All(5) =corr(C6(:),A(:))^2;
R2All(6) =corr(C7(:),A(:))^2;

X=[2 3 4 5 6 7]
plot(X,R2All)


```

- First two lines are just file importing of the dataset.
- At this point the data has been processed as outline [in](./analysis-notes.md)





```python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import NMF

# Load Data
data = pd.read_csv("normalize-turgay.csv")
A = data.to_numpy()

num_components = 7
nmf = NMF(n_components=num_components, init='random', random_state=0)
W = nmf.fit_transform(A)
H = nmf.components_

H_normalized = H / np.sum(H, axis=0)
fig, axs = plt.subplots(num_components, 1, figsize=(6, 3*num_components))
for i in range(num_components):
    axs[i].bar(range(len(H_normalized)), H_normalized[:, i])
    axs[i].set_ylabel('Synergy {}'.format(i+1))

plt.show()


```


```python





```


