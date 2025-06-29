import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_1 = pd.read_csv("C://Users//prann//Downloads//Task1//API_SP.POP.TOTL_DS2_en_csv_v2_127006.csv", skiprows=3)
data_2 = pd.read_csv("C://Users//prann//Downloads//Task1//Metadata_Country_API_SP.POP.TOTL_DS2_en_csv_v2_127006.csv")

data = pd.merge(data_1, data_2, on='Country Code')
data = data[['Country Name', 'Country Code', '2022']].dropna()

top10 = data.sort_values(by='2022', ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x='2022', y='Country Name', data=top10, palette='viridis')
plt.title('Top 10 Most Populated Countries in 2022')
plt.xlabel('Population')
plt.ylabel('Country')
plt.tight_layout()
plt.show()
