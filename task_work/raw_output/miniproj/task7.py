from code_base import load_data, summary_stats
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join
from time import perf_counter as time
import random
import sys
import multiprocessing
from multiprocessing import Pool
from numba import jit

def add_colorbar(im, fig, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label('Temperature')

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
                if interior_mask[i, j]:  # interior_mask is for u[1:-1, 1:-1]
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
    

LOAD_DIR= '/../../dtu/projects/02613_2025/data/modified_swiss_dwellings/'
with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
    building_ids=f.read().splitlines()
if len(sys.argv)<2:
    N=1
else:
    N=int(sys.argv[1])
    
print('Using',N,'floor plan(s), and Numba JIT implementation')

#Take a partition of floorplans, chose to select N random floor plans
random.seed(10)

building_ids = random.sample(building_ids, N)

 #Loadfloorplans
all_u0=np.empty((N, 514,514))
all_interior_mask=np.empty((N,512,512),dtype='bool')
for i,bid in enumerate(building_ids):
    u0,interior_mask =load_data(LOAD_DIR,bid)
    all_u0[i]=u0
    all_interior_mask[i]=interior_mask

#Simulation 
#Run jacobi iterations for each floorplan
MAX_ITER=20_000
ABS_TOL=1e-4 

all_u=np.empty_like(all_u0)

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

print(exec_times)