import pandas as pd

import latstability as ls

wt1nondf = pd.read_csv("./wt_1_non-perturbation.csv")
wt2nondf = pd.read_csv("./wt-2-non-perturbation-all.txt", delimiter=",", header=0)
wt2perdf = pd.read_csv("./wt-2-perturbation-all.txt", delimiter=",", header=0)
wt4nondf = pd.read_csv("./wt_4_non-perturbation.csv")
wt4perdf = pd.read_csv("./wt_4_perturbation.csv")
wt5nondf = pd.read_csv("./wt-5-non-perturbation-all.txt", delimiter=",", header=0)
wt5perdf = pd.read_csv("./wt-5-perturbation-all.txt", delimiter=",", header=0)

# Stance Duration calculation
lhl_st_lengths, lhl_st_timings = ls.stance_duration(
    wt4nondf, swonset_channel="57 lHL swon", swoffset_channel="58 lHL swoff"
)
lfl_st_lengths, lfl_st_timings = ls.stance_duration(
    wt4nondf, swonset_channel="53 lFL swon", swoffset_channel="54 lFL swoff"
)
rhl_st_lengths, rhl_st_timings = ls.stance_duration(
    wt4nondf, swonset_channel="55 rHL swon", swoffset_channel="56 rHL swoff"
)
rfl_st_lengths, rfl_st_timings = ls.stance_duration(wt4nondf)
print(f"Right stance duration {rfl_st_lengths}\n")
print(f"Right stance phase beginning {rfl_st_timings}\n")

# For forelimb
fl_step_widths = ls.step_width(
    wt4nondf, rfl_st_timings, lfl_st_timings, rl_y="35 FRy (cm)", ll_y="33 FLy (cm)"
)
print(fl_step_widths)
hl_step_widths = ls.step_width(
    wt4nondf, rhl_st_timings, lhl_st_timings, rl_y="30 HRy (cm)", ll_y="28 HLy (cm)"
)

print(
    ls.copressure(
        wt1nondf, ds_channel="59 Left DS", hl_channel="28 HLy", fl_channel="33 FLy"
    )
)
