import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data_path =r"C:\Users\LAPTOP GAME VIP\Desktop\INTERLAB\data_13.xlsx"
data = pd.read_excel(data_path)
data.head()

plt.figure(figsize=(8,8))
sns.heatmap(data.corr(method = 'pearson')
 , cmap = 'RdYlGn', vmax = 1, vmin = -1, square=True
 , annot=True, fmt = '.2f', cbar=False)
plt.savefig('seabornPandas1.png', dpi=100)
plt.show()