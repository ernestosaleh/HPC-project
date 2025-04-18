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

def add_colorbar(im, fig, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label('Temperature')

def jacobi(u,interior_mask,max_iter,atol=1e-6):
    u=np.copy(u)
    for i in range(max_iter):
    #Compute average of left,right,up and downneighbors,see eq.(1)
        u_new=0.25*(u[1:-1,:-2]+u[1:-1,2:]+u[:-2,1:-1]+u[2:,1:-1])        
        u_new_interior=u_new[interior_mask]
        delta=np.abs(u[1:-1,1:-1][interior_mask]-u_new_interior).max()
        u[1:-1,1:-1][interior_mask]=u_new_interior
        if delta<atol:
            break
    return u    
    

LOAD_DIR= '/../../dtu/projects/02613_2025/data/modified_swiss_dwellings/'
with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
    building_ids=f.read().splitlines()
if len(sys.argv)<3:
    N=1
    max_n_proc=1
else:
    N=int(sys.argv[1])
    max_n_proc=int(sys.argv[2])
    
print('Using',N,'floor plan(s) and maximum number of',max_n_proc,'processes')

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

def jacobi_wrapper(args):
    u0, interior_mask, max_iter, abs_tol = args
    return jacobi(u0, interior_mask, max_iter, abs_tol)

# Use multiprocessing to parallelize the Jacobi iterations
args_list = [(u0, mask, MAX_ITER, ABS_TOL) for u0, mask in zip(all_u0, all_interior_mask)]

exec_times = {}
print(f"Experiment timing for {N} floor plans with up to {max_n_proc} processes, starting from 1 process")
for n_proc in range(1, max_n_proc + 1):
    with Pool(n_proc) as pool:
        t= time()
        # Submit tasks dynamically using apply_async
        results = [pool.apply_async(jacobi_wrapper, args=(args,)) for args in args_list]
        result = [res.get() for res in results]
        t= time() - t
        exec_times[n_proc] = t

    print(f"Processes: {n_proc} - Execution time: {t:.4f} seconds")

print(exec_times)