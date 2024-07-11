import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("ggplot")


mos_total = pd.read_csv("./mos_limbs_combined_all.csv", header=0)
mos_total = mos_total.drop("Unnamed: 0", axis=1)

rows_with_falls = mos_total[mos_total["MoS"] < 0.0]

print(rows_with_falls.head(5))

fall_counts = np.array([])

wt_fall_count = rows_with_falls["Condition"].value_counts()["WT"]
egr3_fall_count = rows_with_falls["Condition"].value_counts()["Egr3"]
predtx_fall_count = rows_with_falls["Condition"].value_counts()["Pre-DTX"]
postdtx_fall_count = rows_with_falls["Condition"].value_counts()["Post-DTX"]

fall_counts = np.append(fall_counts, wt_fall_count)
fall_counts = np.append(fall_counts, egr3_fall_count)
fall_counts = np.append(fall_counts, predtx_fall_count)
fall_counts = np.append(fall_counts, postdtx_fall_count)


conditions = [
    "WT",
    "Egr3",
    "Pre-DTX",
    "Post-DTX",
]

print(fall_counts)


# We know the two coming from WT is during Perturbation
print("By Perturbation State Breakdown")
predtx_non_fall = (rows_with_falls["Condition"] == "Pre-DTX") & (
    rows_with_falls["Perturbation State"] == "Non-Perturbation"
)
predtx_non_fall = rows_with_falls[~predtx_non_fall]
print(predtx_non_fall)


custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set(
    style="white", font="serif", font_scale=1.7, palette="colorblind", rc=custom_params
)

# plt.title("Theoretical Falls by Condition")
plt.ylabel("# of Falls")
plt.bar(conditions[0], wt_fall_count)
plt.bar(conditions[1], egr3_fall_count)
plt.bar(conditions[2], predtx_fall_count)
plt.bar(conditions[3], postdtx_fall_count)
plt.legend(conditions)

fig = mpl.pyplot.gcf()
fig.set_size_inches(19.8, 10.8)
plt.savefig("./combined_figures/falls_no_title.svg", dpi=300)
# plt.show()
