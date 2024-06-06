import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mos_total = pd.read_csv("./mos_limbs_combined_all.csv", header=0)
mos_total = mos_total.drop("Unnamed: 0", axis=1)

rows_with_falls = mos_total[mos_total["MoS"] < 0.0]

print(rows_with_falls.head(5))

wt_fall_count = rows_with_falls["Condition"].value_counts()["WT"]
egr3_fall_count = rows_with_falls["Condition"].value_counts()["Egr3"]
predtx_fall_count = rows_with_falls["Condition"].value_counts()["Pre-DTX"]
postdtx_fall_count = rows_with_falls["Condition"].value_counts()["Post-DTX"]

print()
print(f"How many theoretical falls for WT {wt_fall_count}")
print(f"How many theoretical falls for Egr3 {egr3_fall_count}")
print(f"How many theoretical falls for Pre-DTX {predtx_fall_count}")
print(f"How many theoretical falls for Post-DTX {postdtx_fall_count}")
print()

# We know the two coming from WT is during Perturbation
print("By Perturbation State Breakdown")
predtx_non_fall = (rows_with_falls['Condition'] == 'Pre-DTX') & (rows_with_falls['Perturbation State'] == 'Non-Perturbation')
predtx_non_fall = rows_with_falls[~predtx_non_fall]
print(predtx_non_fall)

