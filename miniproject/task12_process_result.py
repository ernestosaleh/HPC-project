from code_base import load_data, jacobi, summary_stats
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

data = pd.read_csv('results/task12summary_statistics.csv')

# What is the distribution of the mean temperatures? Show your results as histograms.
fig = data['mean_temp'].plot(kind='hist', bins = 20, edgecolor='black')
plt.xlabel('Mean Temperature')
plt.ylabel('Nr of buildings in bin')
plt.title('Distribution of Mean Temperature')
plt.savefig('images/task12/histogram_mean_temp.png')
plt.close()

# What is the average mean temperature of the buildings?
avg_mean_temp = data['mean_temp'].mean()

# What is the average temperature standard deviation?
avg_std_temp = data['std_temp'].mean()

# How many buildings had at least 50% of their area above 18C?
nr_of_buildings_above18 = (data['pct_above_18'] > 50).sum()

# How many buildings had at least 50% of their area below 15C?
nr_of_buildings_below15 = (data['pct_below_15'] > 50).sum()

#Result
print(f'{"Average mean Temperature":<30}{"Average std":<15}{"pct above 18":<15}{"pct below 15":<15}')
print(f'{avg_mean_temp:<30.2f}{avg_std_temp:<15.2f}{nr_of_buildings_above18:<15}{nr_of_buildings_below15:<15}')
