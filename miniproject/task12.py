from code_base import load_data, jacobi, summary_stats
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from time import perf_counter as time
import random
import multiprocessing
import os
from os.path import join
from multiprocessing import Pool
from numba import jit

from mpl_toolkits.axes_grid1 import make_axes_locatable

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


LOAD_DIR= '/../../dtu/projects/02613_2025/data/modified_swiss_dwellings/'

with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
    building_ids=f.read().splitlines()

if len(sys.argv)<3:
    N=1
    n_proc = 4
else:
    N=int(sys.argv[1])
    n_proc = int(sys.argv[2])

#Loadfloorplans, maybe not load all at once
all_u0=np.empty((N, 514,514))
all_interior_mask=np.empty((N,512,512),dtype='bool')
for i,bid in enumerate(building_ids):
    u0,interior_mask =load_data(LOAD_DIR,bid)
    all_u0[i]=u0
    all_interior_mask[i]=interior_mask


# ######################################################################
# # A very fast implementation of the simulation should be included here
# ######################################################################
MAX_ITER=20_000
ABS_TOL=1e-4 

print('Using',N,'floor plan(s) and ',n_proc,'processors')

@jit(nopython=True)
def jacobi(u, interior_mask, max_iter, atol=1e-6):
    '''
    Thoughts behind: 
    Using numba using for loops often outperforms
    '''
    u = u.copy()
    nrows, ncols = u.shape
    u_new = u.copy()
    for _ in range(max_iter):
        delta = 0.0
        for i in range(1, nrows - 1):
            for j in range(1, ncols - 1):
                if interior_mask[i-1, j-1]:  # interior_mask is for u[1:-1, 1:-1]
                    u_ij_update = 0.25 * (u[i, j - 1] + u[i, j + 1] + u[i - 1, j] + u[i + 1, j]) #Jacobi
                    
                    #compute change
                    d = abs(u[i, j] - u_ij_update)
                    if d > delta: #update delta if the d is larger -> find the largest change and check whether it is within tolerance
                        delta = d
                    u_new[i, j] = u_ij_update
        u = u_new
        if delta < atol:
            break
    return u  

def jacobi_wrapper(args):
    u0, interior_mask, max_iter, abs_tol = args
    return jacobi(u0, interior_mask, max_iter, abs_tol)

# Use multiprocessing to parallelize the Jacobi iterations
args_list = [(u0, mask, MAX_ITER, ABS_TOL) for u0, mask in zip(all_u0, all_interior_mask)]

exec_times = []
all_u=np.empty_like(all_u0)

# Compile function by running it once
jacobi(all_u0[0],all_interior_mask[0],1,ABS_TOL)

for i, (u0,interior_mask, MAX_ITER, ABS_TOL) in enumerate(args_list):   
    t=time()
    u=jacobi(u0,interior_mask,MAX_ITER,ABS_TOL)
    t=time()-t 

    all_u[i]=u
    exec_times.append(t)

#end result
######################################################################
# END OF SIMULATION
######################################################################
######################################################################
# SAVE RESULTS
######################################################################


#Save summary, maybe this should be done when doing the simulation of a floorplan?
SummaryDict = {}
#PrintsummarystatisticsinCSVformat
stat_keys=['mean_temp', 'std_temp','pct_above_18', 'pct_below_15']
for bid,u, interior_mask in zip(building_ids,all_u, all_interior_mask):
    stats=summary_stats(u,interior_mask)
    # Create a dictionary to hold the statistics
    SummaryDict[bid] = stats

# Create a DataFrame from the summary statistics dictionary
summary_df = pd.DataFrame.from_dict(SummaryDict, orient='index')
# Save the summary statistics to a CSV file
summary_df.to_csv('results/task12summary_statistics.csv', index_label='building_id')

#Just in case: save the result in .npz file
#Save the results, in order to visualize them later
np.savez('results/task12_result.npz', exec_times = np.array(exec_times), 
        building_ids = np.array(building_ids), 
        all_u0 = np.array(all_u0),
        all_u = np.array(all_u),
        all_interior_mask = np.array(all_interior_mask))

#Visualize simulations for first 20 floor plans 
for building_id in building_ids[:20]:  
    fig, axes = plt.subplots(1,3, figsize = (10,4), sharey=True)
    visualise_simulation(axes, building_id)
    plt.savefig(f'images/task12/floor_plan_ID{building_id}.png', bbox_inches='tight')
    plt.close()
    
######################################################################
# THE END
######################################################################
