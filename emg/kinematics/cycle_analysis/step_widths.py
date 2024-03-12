import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from statannotations.Annotator import Annotator

import latstability as ls

# Create lists for each cumulative condition
cumulative_flwidth_non = np.empty(0)
cumulative_hlwidth_non = np.empty(0)
cumulative_flwidth_per = np.empty(0)
cumulative_hlwidth_per = np.empty(0)
step_width_results = {}

print("Step Width for M1 without Perturbation")

wt1nondf = pd.read_csv("./wt_1_non-perturbation.csv")

# Getting stance duration for all 4 limbs
lhl_st_lengths, lhl_st_timings = ls.stance_duration(
    wt1nondf, swonset_channel="51 HLl Sw on", swoffset_channel="52 HLl Sw of"
)
lfl_st_lengths, lfl_st_timings = ls.stance_duration(
    wt1nondf, swonset_channel="55 FLl Sw on", swoffset_channel="56 FLl Sw of"
)
rhl_st_lengths, rhl_st_timings = ls.stance_duration(
    wt1nondf, swonset_channel="53 HLr Sw on", swoffset_channel="54 HLr Sw of"
)
rfl_st_lengths, rfl_st_timings = ls.stance_duration(
    wt1nondf, swonset_channel="57 FLr Sw on", swoffset_channel="58 FLr Sw of"
)

# For forelimb
wt1_fl_step_widths = ls.step_width(
    wt1nondf, rfl_st_timings, lfl_st_timings, rl_y="35 FRy", ll_y="33 FLy"
)
wt1_hl_step_widths = ls.step_width(
    wt1nondf, rhl_st_timings, lhl_st_timings, rl_y="30 HRy", ll_y="28 HLy"
)

# Appending values to respective lists
cumulative_flwidth_non = np.concatenate((cumulative_flwidth_non, wt1_fl_step_widths))
cumulative_hlwidth_non = np.concatenate((cumulative_hlwidth_non, wt1_hl_step_widths))

print()
print("Average hindlimb width")
print(np.mean(wt1_hl_step_widths))
print("Average forelimb width")
print(np.mean(wt1_fl_step_widths))
print()

print("Step Width for M1 with Perturbation")

# Example for stance duration based on toex
wt1perdf = pd.read_csv("./wt_1_perturbation-update.csv", header=0)

# Getting stance duration for all 4 limbs
lhl_st_lengths, lhl_st_timings = ls.stance_duration(
    wt1perdf, swonset_channel="51 HLl Sw on", swoffset_channel="52 HLl Sw of"
)
lfl_st_lengths, lfl_st_timings = ls.stance_duration(
    wt1perdf, swonset_channel="55 FLl Sw on", swoffset_channel="56 FLl Sw of"
)
rhl_st_lengths, rhl_st_timings = ls.stance_duration(
    wt1perdf, swonset_channel="53 HLr Sw on", swoffset_channel="54 HLr Sw of"
)
rfl_st_lengths, rfl_st_timings = ls.stance_duration(
    wt1perdf, swonset_channel="57 FLr Sw on", swoffset_channel="58 FLr Sw of"
)

# For forelimb
wt_1_per_fl_step_widths = ls.step_width(
    wt1perdf, rfl_st_timings, lfl_st_timings, rl_y="35 FRy (cm)", ll_y="33 FLy (cm)"
)
wt_1_per_hl_step_widths = ls.step_width(
    wt1perdf, rhl_st_timings, lhl_st_timings, rl_y="30 HRy (cm)", ll_y="28 HLy (cm)"
)

# Appending
cumulative_flwidth_per = np.concatenate(
    (cumulative_flwidth_per, wt_1_per_fl_step_widths)
)
cumulative_hlwidth_per = np.concatenate(
    (cumulative_hlwidth_per, wt_1_per_hl_step_widths)
)

print()
print("Average hindlimb width")
print(np.mean(wt_1_per_hl_step_widths))
print("Average forelimb width")
print(np.mean(wt_1_per_fl_step_widths))
print()

#################################
# Mouse 4
#################################

print("Step Width for M4 without Perturbation")

# Example for stance duration based on toex
wt4nondf = pd.read_csv("./wt_4_non-perturbation.csv")

# Getting stance duration for all 4 limbs
lhl_st_lengths, lhl_st_timings = ls.stance_duration(
    wt4nondf, swonset_channel="57 lHL swon", swoffset_channel="58 lHL swoff"
)
lfl_st_lengths, lfl_st_timings = ls.stance_duration(
    wt4nondf, swonset_channel="53 lFL swon", swoffset_channel="54 lFL swoff"
)
rhl_st_lengths, rhl_st_timings = ls.stance_duration(
    wt4nondf, swonset_channel="55 rHL swon", swoffset_channel="56 rHL swoff"
)
rfl_st_lengths, rfl_st_timings = ls.stance_duration(
    wt4nondf, swonset_channel="51 rFL swon", swoffset_channel="52 rFL swoff"
)

# For forelimb
wt4_fl_step_widths = ls.step_width(
    wt4nondf, rfl_st_timings, lfl_st_timings, rl_y="35 FRy (cm)", ll_y="33 FLy (cm)"
)
wt4_hl_step_widths = ls.step_width(
    wt4nondf, rhl_st_timings, lhl_st_timings, rl_y="30 HRy (cm)", ll_y="28 HLy (cm)"
)

cumulative_flwidth_non = np.concatenate((cumulative_flwidth_non, wt4_fl_step_widths))
cumulative_hlwidth_non = np.concatenate((cumulative_hlwidth_non, wt4_hl_step_widths))

print()
print("Average hindlimb width")
print(np.mean(wt4_hl_step_widths))
print("Average forelimb width")
print(np.mean(wt4_fl_step_widths))
print()

print("Step Width for M4 with Perturbation")
# Example for stance duration based on toex
wt4perdf = pd.read_csv("./wt_4_perturbation.csv")

# Getting stance duration for all 4 limbs
lhl_st_lengths, lhl_st_timings = ls.stance_duration(
    wt4perdf, swonset_channel="57 lHL swon", swoffset_channel="58 lHL swoff"
)
lfl_st_lengths, lfl_st_timings = ls.stance_duration(
    wt4perdf, swonset_channel="53 lFL swon", swoffset_channel="54 lFL swoff"
)
rhl_st_lengths, rhl_st_timings = ls.stance_duration(
    wt4perdf, swonset_channel="55 rHL swon", swoffset_channel="56 rHL swoff"
)
rfl_st_lengths, rfl_st_timings = ls.stance_duration(
    wt4perdf, swonset_channel="51 rFL swon", swoffset_channel="52 rFL swoff"
)

# For forelimb
wt_4_per_fl_step_widths = ls.step_width(
    wt4perdf, rfl_st_timings, lfl_st_timings, rl_y="35 FRy (cm)", ll_y="33 FLy (cm)"
)
wt_4_per_hl_step_widths = ls.step_width(
    wt4perdf, rhl_st_timings, lhl_st_timings, rl_y="30 HRy (cm)", ll_y="28 HLy (cm)"
)

# Concatenating
cumulative_flwidth_per = np.concatenate(
    (cumulative_flwidth_per, wt_4_per_fl_step_widths)
)
cumulative_hlwidth_per = np.concatenate(
    (cumulative_hlwidth_per, wt_4_per_hl_step_widths)
)

print()
print("Average hindlimb width")
print(np.mean(wt_4_per_hl_step_widths))
print("Average forelimb width")
print(np.mean(wt_4_per_fl_step_widths))
print()

# %%
print("Cumulative values")
print()
print("Step widths for non-perturbation")
print("FL:", np.mean(cumulative_flwidth_non))
print()
print("HL:", np.mean(cumulative_hlwidth_non))
print()
print("Step widths for perturbation")
print("FL:", np.mean(cumulative_flwidth_per))
print()
print("HL:", np.mean(cumulative_hlwidth_per))
print()

# %% Saving values
np.savetxt("./flwidths_non.csv", cumulative_flwidth_non, delimiter=",")
np.savetxt("./flwidths_per.csv", cumulative_flwidth_per, delimiter=",")
np.savetxt("./hlwidths_non.csv", cumulative_hlwidth_non, delimiter=",")
np.savetxt("./hlwidths_per.csv", cumulative_hlwidth_per, delimiter=",")

# Saving as a 2-D array

# Saving as a dictionary
step_width_results["Forelimb Non-Perturbation"] = cumulative_flwidth_non
step_width_results["Hindlimb Non-Perturbation"] = cumulative_hlwidth_non
step_width_results["Forelimb Perturbation"] = cumulative_flwidth_per
step_width_results["Hindlimb Perturbation"] = cumulative_hlwidth_per
print(step_width_results)

# Analysis
fl_test = sp.stats.ttest_ind(
    cumulative_flwidth_non, cumulative_flwidth_per, equal_var=False
)
print(f"Comparing Forelimb widths:\n{fl_test}\n")

hl_test = sp.stats.ttest_ind(
    cumulative_hlwidth_non, cumulative_hlwidth_per, equal_var=False
)
print(f"Comparing Hindlimb widths:\n{hl_test}\n")

# Plotting
# sns.barplot(ste)
# plt.show()
# %%
