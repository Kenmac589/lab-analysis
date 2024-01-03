
>>> how can I plot only the numpy indices that are returned from the np.where() function

You can use the `plot()` function provided by the `matplotlib` library to plot the numpy indices returned by the `np.where()` function. Here's an example:
```
import numpy as np
import matplotlib.pyplot as plt

# Generate some random data
data = np.random.rand(10, 10)

# Use np.where() to find the indices of non-zero elements in the array
indices = np.where(data > 0)

# Plot the indices using matplotlib
plt.plot(indices[0], indices[1])
plt.show()
```
This will create a plot that shows the x and y coordinates of the non-zero elements in the array. The `hnp.where()` function returns a tuple containing two arrays, each representing the x and y coordinates of the non-zero elements, respectively. By using the `plot()` function with these arrays as input, you can create a scatter plot that shows the location of the non-zero elements in the array.


```python
import matplotlib.pyplot as plt
import numpy as np

# Create two sample data frames
df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
df2 = pd.DataFrame({'C': [7, 8, 9], 'D': [10, 11, 12]})

# Create a figure and axis object
fig, ax = plt.subplots()

# Plot the data frames on the same axis
ax.bar(df1.index, df1['A'], color='blue')
ax.bar(df2.index, df2['C'], bottom=df1['A'], color='red')

# Set the y-axis limits
ax.set_ylim([0, 15])

# Add a legend to the plot
ax.legend()

# Show the plot
plt.show()
```
This will create a barplot with two bars, one for each data frame. The first bar represents the values in `df1['A']`, and the second bar represents the values in `df2['C']`. The y-axis limits are set to 0
to 15, which is the maximum value of the sum of both data frames.

You can customize the appearance of the plot by using various options available in the `matplotlib` library, such as changing the color palette, adding a title or labels, and adjusting the spacing between
bars.
