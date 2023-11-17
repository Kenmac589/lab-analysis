import numpy as np

# Example 1D NumPy arrays
data1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
data2 = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

# Define the bin ranges
bin_ranges = [0, 3, 6, 10]  # Example bin ranges: [0-3), [3-6), [6-10]

# Compute the bin indices for each element of data
bin_indices = np.digitize(data1, bin_ranges) - 1

# Compute the number of bins
num_bins = len(bin_ranges) - 1

# Sort the indices based on bin indices
sorted_indices = np.argsort(bin_indices)

# Create a new 2D array using the sorted indices
new_array = np.zeros((num_bins, len(data1)))

for i, index in enumerate(sorted_indices):
    new_array[bin_indices[index], i] = data2[index]

# Print the new 2D array
print(new_array)
