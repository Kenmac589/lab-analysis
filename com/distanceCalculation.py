import pandas as pd
import numpy as np

# create a sample pandas DataFrame with x and y columns for each set of coordinates
df = pd.DataFrame({
    'x1': [1, 2, 3],
    'y1': [4, 5, 6],
    'x2': [7, 8, 9],
    'y2': [10, 11, 12]
})

# calculate the difference between the coordinates
dx = df['x2'] - df['x1']
dy = df['y2'] - df['y1']

# calculate the Euclidean distance
distance = np.linalg.norm([dx, dy], axis=0)

# add the distance as a new column in the DataFrame
df['distance'] = distance

print(df)
