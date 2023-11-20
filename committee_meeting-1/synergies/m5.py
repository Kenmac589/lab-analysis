import motorpyrimitives as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import f_oneway


# Pre DTX group initial work
data_selection_non, syn_selection_non = './full_width_test/norm-emg-preDTX-100.csv', 3
motor_p_non, motor_m_non = mp.synergy_extraction(data_selection_non, syn_selection_non)
fwhl_non, fwhl_non_start_stop, fwhl_height_non = mp.full_width_half_abs_min(motor_p_non, syn_selection_non)

data_selection_per, syn_selection_per = './full_width_test/norm-emg-preDTX-per.csv', 3
motor_p_per, motor_m_per = mp.synergy_extraction(data_selection_per, syn_selection_per)
fwhl_per, fwhl_per_start_stop, fwhl_height_per = mp.full_width_half_abs_min(motor_p_per, syn_selection_per)

# Post DTX Group work
data_selection_non_post, syn_selection_non_post = './full_width_test/norm-emg-postDTX-100.csv', 2
motor_p_non_post, motor_m_non_post = mp.synergy_extraction(data_selection_non_post, syn_selection_non_post)
fwhl_non_post, fwhl_non_start_stop_post, fwhl_height_non_post = mp.full_width_half_abs_min(motor_p_non_post, syn_selection_non_post)

# Trace Plots
mp.sel_primitive_trace(data_selection_non, syn_selection_non, "M5 PreDTX Non-pertubation 0.100m/s")
mp.sel_primitive_trace(data_selection_per, syn_selection_per, "M5 PreDTX with Pertubation 0.100m/s")
mp.sel_primitive_trace(data_selection_non_post, syn_selection_non_post, "M5 PostDTX with Pertubation 0.100m/s")


results = dict()
results.update({'PreDTX Non': [np.mean(fwhl_non), np.std(fwhl_non)]})
results.update({'PreDTX Per': [np.mean(fwhl_per), np.std(fwhl_per)]})
results.update({'PostDTX Non': [np.mean(fwhl_non_post), np.std(fwhl_non_post)]})

print(results)

# Comparison

# Comparing predtx conditions
# t_stat, p_value = ttest_ind(fwhl_non, fwhl_per)
# print("T-test between predtx conditions", t_stat, p_value)
#
# t_stat, p_value = ttest_ind("T-test between predtx and post", fwhl_non, fwhl_non_post)
# print(t_stat, p_value)


# Plotting
trials = list(results.keys())
print(trials)

mean_fwhl = [value[0] for value in results.values()]
sd_fwhl = [value[1] for value in results.values()]
plt.title('Average full width half maximum length')
plt.bar(trials, mean_fwhl, yerr=sd_fwhl, capsize=5, align='center', alpha=0.6)
plt.savefig('../figures/fwhl_comparison.png', dpi=300)
plt.show()
