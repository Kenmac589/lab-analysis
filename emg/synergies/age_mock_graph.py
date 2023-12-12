import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# create DataFrame
df = pd.DataFrame({
    'Day': ['WT', 'PreDTX', 'PostDTX', 'WT', 'PreDTX', 'PostDTX'],
    'Customers': [44, 46, 49, 59, 54, 33, 46, 50, 49, 60],
    'Conditions': ['Non-Perturbation', 'Non-Perturbation', 'Non-Perturbation', 'Perturbation', 'Perturbation', 'Perturbation']
})

# set seaborn plotting aesthetics
sns.set(style='white')

# create grouped bar chart
sns.barplot(x='Day', y='Customers', hue='Time', data=df)
plt.show()
