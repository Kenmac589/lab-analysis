import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import latstability as ls

print("Step Width for M1 without Perturbation")

wt1nondf = pd.read_csv('./wt_1_non-perturbation.csv')

# Getting stance duration for all 4 limbs
lhl_st_lengths, lhl_st_timings = ls.stance_duration(wt1nondf, swonset_channel="51 HLl Sw on", swoffset_channel="52 HLl Sw of")
lfl_st_lengths, lfl_st_timings = ls.stance_duration(wt1nondf, swonset_channel="55 FLl Sw on", swoffset_channel="56 FLl Sw of")
rhl_st_lengths, rhl_st_timings = ls.stance_duration(wt1nondf, swonset_channel="53 HLr Sw on", swoffset_channel="54 HLr Sw of")
rfl_st_lengths, rfl_st_timings = ls.stance_duration(wt1nondf, swonset_channel="57 FLr Sw on", swoffset_channel="58 FLr Sw of")

# For forelimb
wt1_fl_step_widths = ls.step_width(wt1nondf, rfl_st_timings, lfl_st_timings, rl_y="35 FRy", ll_y="33 FLy")
wt1_hl_step_widths = ls.step_width(wt1nondf, rhl_st_timings, lhl_st_timings, rl_y="30 HRy", ll_y="28 HLy")

print()
print("Average hindlimb width")
print(np.mean(wt1_hl_step_widths))
print("Average forelimb width")
print(np.mean(wt1_fl_step_widths))
print()

print("Step Width for M1 with Perturbation")

# Example for stance duration based on toex
wt1perdf = pd.read_csv('./wt_1_perturbation.csv', header=0)

# Getting stance duration for all 4 limbs
lhl_st_lengths, lhl_st_timings = ls.stance_duration(wt1perdf, swonset_channel="51 HLl Sw on", swoffset_channel="52 HLl Sw of")
lfl_st_lengths, lfl_st_timings = ls.stance_duration(wt1perdf, swonset_channel="55 FLl Sw on", swoffset_channel="56 FLl Sw of")
rhl_st_lengths, rhl_st_timings = ls.stance_duration(wt1perdf, swonset_channel="53 HLr Sw on", swoffset_channel="54 HLr Sw of")
rfl_st_lengths, rfl_st_timings = ls.stance_duration(wt1perdf, swonset_channel="57 FLr Sw on", swoffset_channel="58 FLr Sw of")

# For forelimb
fl_step_widths = ls.step_width(wt1perdf, rfl_st_timings, lfl_st_timings, rl_y="35 FRy", ll_y="33 FLy")
hl_step_widths = ls.step_width(wt1perdf, rhl_st_timings, lhl_st_timings, rl_y="30 HRy", ll_y="28 HLy")

print()
print("Average hindlimb width")
print(np.mean(hl_step_widths))
print("Average forelimb width")
print(np.mean(fl_step_widths))
print()


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
