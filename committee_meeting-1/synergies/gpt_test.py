import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data into two separate dataframes
df1 = pd.read_csv("dataframe1.csv")
df2 = pd.read_csv("dataframe2.csv")

# Align the indices of both data frames
df1 = df1.set_index(["column1", "column2"])
df2 = df2.set_index(["column1", "column2"])

# Create a paired barplot using seaborn's `barplot` function
sns.barplot(x="column3", y="column4", data=df1, color="blue")
sns.barplot(x="column5", y="column6", data=df2, color="red")

# Add a legend to the plot
plt.legend()

# Show the plot
plt.show()
