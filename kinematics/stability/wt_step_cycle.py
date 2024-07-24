import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import latstability as ls

# from statannotations.Annotator import Annotator


print("Step Width for M1 without Perturbation")

wt1nondf = pd.read_csv("./wt_data/wt-1-non-all.txt")

# Getting stance duration for all 4 limbs

wt1_swingon, wt1_swingoff = ls.swing_estimation(wt1nondf, x_channel="32 FLx (cm)")
wt1_right_ds = ls.double_support_est(
    wt1nondf, fl_channel="34 FRx (cm)", hl_channel="29 HRx (cm)", manual_peaks=False
)

x_cord = wt1nondf["32 FLx (cm)"].to_numpy(dtype=float)

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set(style="white", font_scale=1.0, rc=custom_params)

swing_legend = [
    "Limb X cord",
    "Swing offset",
    "Swing onset",
]

# For plotting figure demonstrating how calculation was done
plt.title("Swing Estimation")
plt.plot(x_cord)
plt.plot(wt1_swingoff, x_cord[wt1_swingoff], "^")
plt.plot(wt1_swingon, x_cord[wt1_swingon], "v")
plt.legend(swing_legend, bbox_to_anchor=(1, 0.7))

# Looking at result
# axs[1].set_title("MoS Result")
# axs[1].bar(0, np.mean(lmos), yerr=np.std(lmos), capsize=5)
# axs[1].bar(1, np.mean(rmos), yerr=np.std(rmos), capsize=5)
# axs[1].legend(mos_legend, bbox_to_anchor=(1, 0.7))

plt.tight_layout()
plt.show()
