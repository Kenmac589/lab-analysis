import pandas as pd

import latstability as ls

# Pre-DTX Mice here
# Need to get hip heights for xcom
# dtrpre_2non = pd.read_csv(
#     "./dtr_data/predtx/dtr-pre-2-non-all.txt", delimiter=",", header=0
# )
# dtrpre_2per = pd.read_csv(
#     "./dtr_data/predtx/dtr-pre-2-per-all.txt", delimiter=",", header=0
# )
# dtrpre_2sin = pd.read_csv(
#     "./dtr_data/predtx/dtr-pre-2-sin-all.txt", delimiter=",", header=0
# )
#
# dtrpre_3non = pd.read_csv(
#     "./dtr_data/predtx/dtr-pre-3-non-all.txt", delimiter=",", header=0
# )
# dtrpre_3per = pd.read_csv(
#     "./dtr_data/predtx/dtr-pre-3-per-all.txt", delimiter=",", header=0
# )
# dtrpre_3sin = pd.read_csv(
#     "./dtr_data/predtx/dtr-pre-3-sin-all-2.txt", delimiter=",", header=0
# )
# dtrpre_3non = pd.read_csv(
#     "./dtr_data/predtx/dtr-pre-3-non-all.txt", delimiter=",", header=0
# )
# dtrpre_5non = pd.read_csv(
#     "./dtr_data/predtx/dtr-pre-5-non-all.txt", delimiter=",", header=0
# )
# dtrpre_5per = pd.read_csv(
#     "./dtr_data/predtx/dtr-pre-5-per-all-2.txt", delimiter=",", header=0
# )
# dtrpre_5sin = pd.read_csv(
#     "./dtr_data/predtx/dtr-pre-5-sin-all.txt", delimiter=",", header=0
# )


# Post-DTX Mice
dtr_post_5non = pd.read_csv(
    "./dtr_data/postdtx/hip_height_calcs/dtr-post-5-non-hip-data.txt",
    delimiter=",",
    header=0,
)
dtr_post_5per = pd.read_csv(
    "./dtr_data/postdtx/hip_height_calcs/dlc-post-5-per-hip-data.txt",
    delimiter=",",
    header=0,
)
dtr_post_5sin = pd.read_csv(
    "./dtr_data/postdtx/hip_height_calcs/dtr-post-5-sin-hip-data.txt",
    delimiter=",",
    header=0,
)

# Hip height for postdtx
dtr_post_5non_hiph = ls.hip_height(
    dtr_post_5non, toey="25 toey (cm)", hipy="17 Hipy (cm)", manual=False
)
dtr_post_5per_hiph = ls.hip_height(
    dtr_post_5per, toey="25 toey (cm)", hipy="17 Hipy (cm)", manual=False
)
dtr_post_5sin_hiph = ls.hip_height(
    dtr_post_5sin, toey="25 toey (cm)", hipy="17 Hipy (cm)", manual=True
)

print(f"DTR M5 non Hip automatically {dtr_post_5non_hiph}")
print(f"DTR M5 per Hip automatically {dtr_post_5per_hiph}")
print(f"DTR M5 sin Hip manually {dtr_post_5sin_hiph}")
