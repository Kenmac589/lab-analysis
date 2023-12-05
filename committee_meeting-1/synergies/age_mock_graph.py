import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Plotting
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set(style="white", rc=custom_params)

# data_path = sys.argv[1]
example_data = {
    '4mo Non': [0.3, 0.1], '4mo Per': [0.6, 0.1],
    '1yr Non': [0.5, 0.1], '1yr Per': [0.8, 0.1],
    '2yr Non': [0.9, 0.1], '2yr Per': [0.9, 0.1]
}



# Read in the frame
# data_input = pd.read_csv(data_path, header=0)

# data_input.plot(subplots=True)
plt.ylim(0, 2.0)
plt.bar(*zip(*example_data.items()))
plt.show()
