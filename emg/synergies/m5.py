import motorpyrimitives as mp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
from scipy import stats as st
mp.nnmf_factorize

def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    """
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y + barh, y + barh, y]
    mid = ((lx + rx) / 2, y + barh)

    plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)

# Main work
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
mp.sel_primitive_trace(data_selection_non, syn_selection_non, "M5 PreDTX Non-pertubation Synergy {} at 0.100m/s".format(syn_selection_non))
mp.sel_primitive_trace(data_selection_per, syn_selection_per, "M5 PreDTX with pertubation Synergy {} at 0.100m/s".format(syn_selection_per))
mp.sel_primitive_trace(data_selection_non_post, syn_selection_non_post, "M5 PostDTX Non-pertubation Synergy {} at 0.100m/s".format(syn_selection_non_post))
# set_title('Synergy {}'.format(col+1))

results = dict()
results.update({'PreDTX Non': [np.mean(fwhl_non), np.std(fwhl_non)]})
results.update({'PreDTX Per': [np.mean(fwhl_per), np.std(fwhl_per)]})
results.update({'PostDTX Non': [np.mean(fwhl_non_post), np.std(fwhl_non_post)]})

print(results)

# Comparison

# Comparing effect of pertubation in preDTX

t_stat, p_value_per = ttest_ind(fwhl_non, fwhl_per)
print("T-test between predtx conditions", t_stat, p_value_per)

# Comparing effect of Diptheria injection without Perturbation
t_stat, p_value_dtx = ttest_ind(fwhl_non, fwhl_non_post)
print("T-test between pre and post diptheria injection", t_stat, p_value_dtx)



# Plotting
sns.set()
trials = list(results.keys())
print(trials)

mean_fwhl = [value[0] for value in results.values()]
sd_fwhl = [value[1] for value in results.values()]
print(sd_fwhl)

height = mean_fwhl
bars = np.arange(len(mean_fwhl))
fig, ax = plt.subplots()
plt.title('Average full width half maximum length for 3 Synergies')
# plt.errorbar(trials, mean_fwhl, yerr=sd_fwhl, capsize=3, lolims=False, fmt='.k')
ax = sns.barplot(results, x=results[0], errorbar=results[1])
# plt.bar(trials, mean_fwhl, align='center', alpha=0.6)
# barplot_annotate_brackets(0, 1, p_value_per, bars, height)
# barplot_annotate_brackets(0, 2, p_value_dtx, bars, height)
# # plt.savefig('../figures/fwhl_comparison.png', dpi=300)
plt.show()
