import os

path = '/Users/kenzie_mackinnon/sync/lab-analysis/emg/CoM-M3-WT-20220220'

for i in range(1, 26):
    path_cand = path + f"/CoM-M3-WT-20220420-{i}"
    if not os.path.exists(path_cand):
        os.mkdir(path_cand)
