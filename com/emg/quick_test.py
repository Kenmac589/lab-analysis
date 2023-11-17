import numpy as np
import matplotlib.pyplot as plt

# Create multiple NumPy arrays for the data
data1 = np.array([1, 2, 3, 4, 5])
data2 = np.array([2, 4, 6, 8, 10])
data3 = np.array([3, 6, 9, 12, 15])

# Set the x-coordinates for the bars
x = np.arange(len(data1))

# Calculate the means and standard deviations for each dataset
mean1 = np.mean(data1)
std1 = np.std(data1)

mean2 = np.mean(data2)
std2 = np.std(data2)

mean3 = np.mean(data3)
std3 = np.std(data3)

# Plot the bar graph
plt.bar(x, [mean1, mean2, mean3], tick_label=['Data 1', 'Data 2', 'Data 3'])

# Plot the error bars without the upper limit arrow
plt.errorbar(x, [mean1, mean2, mean3], yerr=[std1, std2, std3], fmt='o', color='red', capsize=0)

# Manually draw the upper error bars
plt.plot(x, [mean1+std1, mean2+std2, mean3+std3], '_', color='red')

# Set labels and title for the graph
plt.xlabel('Data')
plt.ylabel('Mean Values')
plt.title('Bar Graph with Error Bars')

# Display the plot
plt.show()
