import pandas as pd

import latstability as ls

# For Egr3 KO's
egr3_6nondf = pd.read_csv(
    "./egr3_cop_mistake/egr3-6-non-perturbation-all.txt", delimiter=",", header=0
)
egr3_7nondf = pd.read_csv(
    "./egr3_cop_mistake/egr3-7-non-perturbation-all.txt", delimiter=",", header=0
)
egr3_8nondf = pd.read_csv(
    "./egr3_cop_mistake/egr3-8-non-perturbation-all.txt", delimiter=",", header=0
)
egr3_9nondf = pd.read_csv(
    "./egr3_cop_mistake/egr3-9-non-perturbation-all.txt", delimiter=",", header=0
)
egr3_10nondf = pd.read_csv(
    "./egr3_cop_mistake/egr3-10-non-perturbation-all.txt", delimiter=",", header=0
)
egr3_8perdf = pd.read_csv(
    "./egr3_cop_mistake/egr3-8-perturbation-all.txt", delimiter=",", header=0
)


egr3_6non_hip_h = ls.hip_height(
    egr3_6nondf, toey="24 toey (cm)", hipy="16 Hipy (cm)", manual=False
)
egr3_7non_hip_h = ls.hip_height(
    egr3_7nondf, toey="24 toey (cm)", hipy="16 Hipy (cm)", manual=True
)
egr3_8non_hip_h = ls.hip_height(
    egr3_8nondf, toey="24 toey (cm)", hipy="16 Hipy (cm)", manual=False
)
egr3_9non_hip_h = ls.hip_height(
    egr3_9nondf, toey="24 toey (cm)", hipy="16 Hipy (cm)", manual=False
)
egr3_8perdf_hip_h = ls.hip_height(egr3_8perdf)
egr3_10non_hip_h = ls.hip_height(
    egr3_10nondf, toey="24 toey (cm)", hipy="16 Hipy (cm)", manual=True
)

print(f"Egr3 M6  Hip manually {egr3_6non_hip_h}")
print(f"Egr3 M7  Hip manually {egr3_7non_hip_h}")
print(f"Egr3 M8  Hip manually {egr3_8non_hip_h}")
print(f"Egr3 M8 Hip {egr3_8perdf_hip_h}")
print(f"Egr3 M9  Hip manually {egr3_9non_hip_h}")
print(f"Egr3 M10 Hip manually {egr3_10non_hip_h}")
