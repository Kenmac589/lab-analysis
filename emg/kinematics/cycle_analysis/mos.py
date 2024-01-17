import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import latstability as ls

wt1nondf = pd.read_csv('./wt_1_non-perturbation.csv', header=0)
wt_1_non_step_cycles = ls.extract_cycles(wt1nondf)
wt1perdf = pd.read_csv('./wt_1_perturbation.csv', header=0)
wt_1_per_step_cycles = ls.extract_cycles(wt1perdf)
# np.savetxt(wt_1_step_cycles, './wt_1_non-perturbation_step_cycles.csv', delimiter=',')
