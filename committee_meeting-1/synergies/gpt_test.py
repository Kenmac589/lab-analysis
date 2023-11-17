import pandas as pd

# Example DataFrame
data = {
    'start_column': [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    'end_column': [0, 0, 0, 0, 0, 1, 0, 0, 1, 0]
}

df = pd.DataFrame(data)

# Marking the beginning and end of regions
start_indices = df.index[df['start_column'] == 1].tolist()
end_indices = df.index[df['end_column'] == 1].tolist()

for start_idx in start_indices:
    for end_idx in end_indices:
        if end_idx > start_idx:
            df.loc[start_idx, 'Start_of_Region'] = 1
            df.loc[end_idx, 'End_of_Region'] = 1

# Forward fill to propagate the labels within the regions
df['Start_of_Region'].fillna(method='ffill', inplace=True)
df['End_of_Region'].fillna(method='bfill', inplace=True)

# Display the modified DataFrame
print(df)
