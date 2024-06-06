import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statannotations import Annotator

import latstability as ls

wt1nondf = pd.read_csv("./wt_data/wt_1_perturbation.csv", delimiter=",", header=0)
wt1perdf = pd.read_csv("./wt_data/wt-1_perturbation-all.txt", delimiter=",", header=0)
wt1sindf = pd.read_csv("./wt_data/wt-1-sinus-all.txt", delimiter=",", header=0)
wt2nondf = pd.read_csv("./wt_data/wt-2-non-perturbation-all.txt", delimiter=",", header=0)
wt2perdf = pd.read_csv("./wt_data/wt-2-perturbation-all.txt", delimiter=",", header=0)
wt2sindf = pd.read_csv("./wt_data/wt-2-sinus-all.txt", delimiter=",", header=0)
wt3nondf = pd.read_csv("./wt_data/wt-3-non-perturbation.txt", delimiter=",", header=0)
wt3perdf = pd.read_csv("./wt_data/", delimiter=",", header=0)
wt3nondf = pd.read_csv("./wt_data/wt-3-non-perturbation.txt", delimiter=",", header=0)

