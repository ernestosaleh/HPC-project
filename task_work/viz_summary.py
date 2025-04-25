from code_base import load_data, jacobi, summary_stats
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

data = pd.read_csv('results/task1_to_3summary_statistics.csv')

# What is the distribution of the mean temperatures? Show your results as histograms.
fig = data['mean_temp'].plot(kind='hist', bins = 20, edgecolor='black')
plt.xlabel('Mean Temperature')
plt.ylabel('Nr of buildings in bin')
plt.title('Distribution of Mean Temperature')


# What is the average mean temperature of the buildings?
print(data['mean_temp'].mean())

# What is the average temperature standard deviation?
print(data['std_temp'].mean())

# How many buildings had at least 50% of their area above 18C?
print((data['pct_above_18'] > 50).sum())

# How many buildings had at least 50% of their area below 15C?
print((data['pct_below_15'] > 50).sum())

plt.show()