import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random

df = pd.read_csv("C://Users//prann//Downloads//Task4//us_traffic_accidents_2023.csv")

df['TotalIncidents'] = df['NumberOfAccidents'] + df['Fatalities'] + df['SeriousInjuries']

df['Season'] = df['Month'].apply(lambda x: 'Winter' if x in [12, 1, 2] else 'Spring' if x in [3, 4, 5] else 'Summer' if x in [6, 7, 8] else 'Fall')
seasonal_patterns = df.groupby('Season')['NumberOfAccidents'].mean().reset_index()

hotspots = df.groupby('State')['TotalIncidents'].sum().sort_values(ascending=False).head(5)

contributing_factors = df[['NumberOfAccidents', 'Fatalities', 'SeriousInjuries']].corr()

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.barplot(x='Season', y='NumberOfAccidents', data=seasonal_patterns)
plt.title('Average Accidents by Season')
plt.xlabel('Season')
plt.ylabel('Average Number of Accidents')

plt.subplot(2, 2, 2)
sns.barplot(x=hotspots.index, y=hotspots.values)
plt.title('Top 5 Accident Hotspots by Total Incidents')
plt.xlabel('State')
plt.ylabel('Total Incidents')
plt.xticks(rotation=45)

plt.subplot(2, 2, 3)
sns.heatmap(contributing_factors, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation of Contributing Factors')

plt.subplot(2, 2, 4)
sns.scatterplot(x='Month', y='NumberOfAccidents', hue='State', data=df, palette='viridis', alpha=0.6, s=100)
plt.title('Accidents by Month')
plt.xlabel('Month')
plt.ylabel('Number of Accidents')

plt.tight_layout()
plt.show()
