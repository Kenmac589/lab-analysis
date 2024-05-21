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
# dtrpre_6non = pd.read_csv(
#     "./dtr_data/predtx/predtx_hip_height_calcs/dtr-pre-6-non-hip.txt",
#     delimiter=",",
#     header=0,
# )
# dtrpre_6per = pd.read_csv(
#     "./dtr_data/predtx/predtx_hip_height_calcs/dtr-pre-6-per-hip.txt",
#     delimiter=",",
#     header=0,
# )
# dtrpre_6sin = pd.read_csv(
#     "./dtr_data/predtx/predtx_hip_height_calcs/dtr-pre-6-sin-hip.txt",
#     delimiter=",",
#     header=0,
# )
dtrpre_7non = pd.read_csv(
    "./dtr_data/predtx/predtx_hip_height_calcs/dtr-pre-7-non-hip.txt",
    delimiter=",",
    header=0,
)
dtrpre_7per = pd.read_csv(
    "./dtr_data/predtx/predtx_hip_height_calcs/dtr-pre-7-per-hip.txt",
    delimiter=",",
    header=0,
)
dtrpre_7sin = pd.read_csv(
    "./dtr_data/predtx/predtx_hip_height_calcs/dtr-pre-7-sin-hip.txt",
    delimiter=",",
    header=0,
)


# Post-DTX Mice
# dtr_post_non = pd.read_csv(
#     "./dtr_data/postdtx/postdtx_hip_height_calcs/dtr-post-3-non-hip-data.txt",
#     delimiter=",",
#     header=0,
# )
# dtr_post_per = pd.read_csv(
#     "./dtr_data/postdtx/hip_height_calcs/dlc-post-5-per-hip-data.txt",
#     delimiter=",",
#     header=0,
# )
# dtr_post_sin = pd.read_csv(
#     "./dtr_data/postdtx/hip_height_calcs/dtr-post-5-sin-hip-data.txt",
#     delimiter=",",
#     header=0,
# )


# Hip height for postdtx
dtr_pre_non_hiph = ls.hip_height(
    dtrpre_7non, toey="25 toey (cm)", hipy="17 Hipy (cm)", manual=False
)
dtr_pre_per_hiph = ls.hip_height(
    dtrpre_7per, toey="25 toey (cm)", hipy="17 Hipy (cm)", manual=False
)
dtr_pre_sin_hiph = ls.hip_height(
    dtrpre_7sin, toey="25 toey (cm)", hipy="17 Hipy (cm)", manual=False
)

print(f"DTR M7 non Hip automatically {dtr_pre_non_hiph}")
print(f"DTR M7 per Hip automatically {dtr_pre_per_hiph}")
print(f"DTR M7 sin Hip automatically {dtr_pre_sin_hiph}")
