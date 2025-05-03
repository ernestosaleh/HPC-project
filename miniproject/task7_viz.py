from code_base import load_data, jacobi, summary_stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from os.path import join
from time import perf_counter as time
import random

from mpl_toolkits.axes_grid1 import make_axes_locatable

#load data 
data = np.load('results/task7_result.npz', allow_pickle=True)
exec_times = data['exec_times']
building_ids = [str(id) for id in data['building_ids']]
all_u0 = data['all_u0']
all_u = data['all_u']
all_interior_mask = data['all_interior_mask']


def add_colorbar(im, fig, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label('Temperature')
    
def visualise_simulation(ax, building_id):
    idx = building_ids.index(building_id)
    
    axes[0].imshow(all_interior_mask[idx], cmap='grey')
    axes[1].imshow(all_u0[idx], cmap='magma')
    im = axes[2].imshow(all_u[idx], cmap='magma')
    
    axes[0].set(title='Floor Plan')
    axes[1].set(title=f'ID: {building_id}\n\nInitial Condition')
    axes[2].set(title='Heat Distribution')
    add_colorbar(im, fig, axes[2])
    # fig.suptitle(f'ID: {building_id}')   

#Visualize simulations for floor plans in task1_to_3_result.npz
for building_id in building_ids:  
    fig, axes = plt.subplots(1,3, figsize = (10,4), sharey=True)
    visualise_simulation(axes, building_id)
    plt.savefig(f'images/task7/floor_plan_ID{building_id}.png', bbox_inches='tight')
    plt.close()
    
#Execution time plot
fig, ax = plt.subplots(1,1, figsize = (10,4))
ax.bar(building_ids, exec_times)
ax.set(title=f'Execution Times, total: {exec_times.sum():.4f} s', xlabel='Building ID', ylabel='Time (s)')
ax.axhline(y=np.mean(exec_times), color='red', linestyle='--', linewidth=1)  # Add horizontal line
ax.text(18, np.mean(exec_times) + 0.1, f"Average: {np.mean(exec_times):.4f}", color='red', ha='center')
ax.set_xticks(list(range(len(building_ids))))
ax.set_xticklabels(building_ids, rotation=90)
plt.savefig(f'images/task7/execution_time.png', bbox_inches='tight')
plt.close()