import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Plotting
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set(style="white", rc=custom_params)

# data_path = sys.argv[1]
example_data = dict()
example_data.update({'4mo Non': [0.3, 0.1]})
example_data.update({'4mo Per': [0.6, 0.1]})
example_data.update({'1yr Non': [0.65, 0.1]})
example_data.update({'1yr Per': [0.8, 0.1]})
example_data.update({'2yr Non': [0.9, 0.1]})
example_data.update({'2yr Per': [0.9, 0.1]})

trials = list(example_data.keys())
mean_fhwm = [value[0] for value in example_data.values()]
std_fhwm = [value[1] for value in example_data.values()]

plt.title("Some Metric")
plt.ylim(0, 2.0)
mock = sns.barplot(
    x=trials,
    y=mean_fhwm,
    zorder=2
)
mock.set_yticks([])
mock.errorbar(x=trials, y=mean_fhwm, yerr=std_fhwm, capsize=3, fmt="none", c="k", zorder=1)
plt.show()
