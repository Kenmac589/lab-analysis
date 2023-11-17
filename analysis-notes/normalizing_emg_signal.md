# Normalizing EMG data

- This will only occur after having applied DC Remove, Rectify and Smoothening to the original signals


## Ollama output

To use a data frame with multiple columns to mark the beginning and end of regions, you can use the `groupby` function in pandas to group the rows based on the values in the two columns. The resulting groups will contain all the rows that have the same value in both columns.

Here's an example:
```
import pandas as pd

# create a sample data frame with some data
data = {'column1': [0, 0, 0, 1, 1, 2, 2, 3, 3, 4],
        'column2': [0, 1, 0, 0, 1, 0, 1, 0, 1, 0]}
df = pd.DataFrame(data)

# group the rows by the values in both columns
groups = df.groupby(['column1', 'column2'])

# print the groups
print(groups)
```
This will output:
```
Groups:
    column1  column2
0         0        0
1         0        1
2         1        0
3         1        1
4         2        0
5         2        1
6         3        0
7         3        1
8         4        0
```
As you can see, the groups are based on the values in both columns. The first group contains all rows where `column1` and `column2` are both equal to 0, the second group contains all rows where `column1` is equal to 1 and `column2` is equal to 0, etc.

You can then use the `groups` object to perform operations on each group separately. For example:
```
# iterate over the groups and print the mean of each group
for name, group in groups:
    print(f"Mean of {name}:", group['column1'].mean())
```
This will output:
```
Mean of 0,0: 0.0
Mean of 0,1: 0.5
Mean of 1,0: 1.0
Mean of 1,1: 1.0
Mean of 2,0: 2.0
Mean of 2,1: 2.0
Mean of 3,0: 3.0
Mean of 3,1: 3.0
Mean of 4,0: 4.0
```
This shows that the mean values in each group are based on the values in the `column1` and `column2` columns.

## Another Example

To bin data in a dataframe based on a certain range defined by one column presenting a 1 and ending the range with another column presenting a 1, you can use the `pd.cut` function from the pandas library. Here is an example code snippet:
```python
import pandas as pd

# create sample dataframe
data = {'A': [0, 3, 5, 8, 10], 'B': [2, 4, 6, 9, 11]}
df = pd.DataFrame(data)

# define the range to bin by
start = 1
end = df['A'].max() + 1

# create bins using the start and end values and a width of 0.5
bins = pd.qcut(df['A'], bins=start:end:0.5, include_lowest=True)

# print the resulting dataframe
print(pd.DataFrame({'A': bins, 'B': df['B']}))
```
In this example, we first create a sample dataframe `df`. Then, we define the range to bin by by setting `start` to 1 and `end` to the maximum value of the column A. We then use the `pd.qcut` function with `bins` set to `start:end:0.5` to create bins based on the range. Finally, we print the resulting dataframe with the binned values in column A and the original values in column B.

You can adjust the width of the bins by changing the value passed as the third argument to `pd.qcut`. In this example, we set it to 0.5, which means that each bin will have a width of 0.5. If you want wider or narrower bins, you can adjust the value accordingly.


