import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# skip first column: it will not be used for the prediction
columns = ['Cruise_line', 'Age', 'Tonnage', 'passengers', 'length', 'cabins', 'passenger_density', 'crew']

# read csv file and import data in pandas dataframe
df = pd.read_csv('../dataset/dataset.csv', sep=',', usecols=columns)

# convert text value in integer
df['class_int'] = pd.Categorical(df['Cruise_line']).codes

# print head of the file
print(df.head())

df.isnull().sum()

# remove rows that contain missing values
df = df.dropna(axis=0)
df.isnull().sum()

# draw data in order to find linear correlation
#cols1 = ['Cruise_line', 'Age', 'Tonnage', 'passengers', 'crew']
#cols2 = ['length', 'cabins', 'passenger_density', 'crew']

# select columns correlated to crew data
cols = ['Tonnage','passengers','length','cabins']

# plot data
sns.set(style='whitegrid', context='notebook')
sns.pairplot(df[cols], height=2.5)
plt.tight_layout()
plt.show()

# plot covariance matrix
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)

plt.tight_layout()
plt.show()

# result: it seems that passengers and crew columns have linear correlation