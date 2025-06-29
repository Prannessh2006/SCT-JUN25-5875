import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

df = pd.read_csv("train.csv")

print("Shape of data:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)

print("\nSurvival Counts:\n", df['Survived'].value_counts())

sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.xticks([0, 1], ['Not Survived', 'Survived'])
plt.show()

sns.countplot(x='Sex', data=df)
plt.title("Gender Distribution")
plt.show()

sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.legend(labels=["Not Survived", "Survived"])
plt.show()

sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival by Passenger Class")
plt.legend(labels=["Not Survived", "Survived"])
plt.show()

sns.histplot(df['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

sns.boxplot(x='Survived', y='Age', data=df)
plt.title("Age vs Survival")
plt.xticks([0, 1], ['Not Survived', 'Survived'])
plt.show()
