import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import latstability as ls

# wt1nondf = pd.read_csv('./wt_1_non-perturbation.csv', header=0)
# wt_1_non_step_cycles = extract_cycles(wt1nondf)

# hipH = hip_height(wt1nondf, toey="24 toey", hipy="16 Hipy")

# wt1perdf = pd.read_csv('./wt_1_perturbation.csv', header=0)
# wt_1_per_step_cycles = extract_cycles(wt1perdf)

# hipH = hip_height(wt1perdf)
# xcomwtper = xcom(wt1perdf, hipH)

# Example for stance duration based on toex
wt4nondf = pd.read_csv('./wt_4_non-perturbation.csv')

# Getting stance duration for all 4 limbs
lhl_st_lengths, lhl_st_timings = ls.stance_duration(wt4nondf, swonset_channel="57 lHL swon", swoffset_channel="58 lHL swoff")
lfl_st_lengths, lfl_st_timings = ls.stance_duration(wt4nondf, swonset_channel="53 lFL swon", swoffset_channel="54 lFL swoff")
rhl_st_lengths, rhl_st_timings = ls.stance_duration(wt4nondf, swonset_channel="55 rHL swon", swoffset_channel="56 rHL swoff")
rfl_st_lengths, rfl_st_timings = ls.stance_duration(wt4nondf, swonset_channel="51 rFL swon", swoffset_channel="52 rFL swoff")

# For forelimb
print("Forelimb measurements")
fl_step_widths = ls.step_width(wt4nondf, rfl_st_timings, lfl_st_timings, rl_y="35 FRy (cm)", ll_y="33 FLy (cm)")
print()
print("Hindlimb measurements")
hl_step_widths = ls.step_width(wt4nondf, rhl_st_timings, lhl_st_timings, rl_y="30 HRy (cm)", ll_y="28 HLy (cm)")

print()
print("Average hindlimb width")
print(np.mean(hl_step_widths))
print("Average forelimb width")
print(np.mean(fl_step_widths))
