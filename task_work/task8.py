from code_base import load_data, summary_stats
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join
from time import perf_counter as time
import random
import pandas as pd
import sys
import multiprocessing
from multiprocessing import Pool
from numba import jit, cuda

LOAD_DIR= '/../../dtu/projects/02613_2025/data/modified_swiss_dwellings/'
with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
    building_ids=f.read().splitlines()
if len(sys.argv)<2:
    N=1
else:
    N=int(sys.argv[1])
    
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
MAX_ITER=40_000
ABS_TOL=1e-4 

#What do I want the kernel to do? Run the Jacobi simulation untill reaching MAX_EPOCHS
# Each output is 514 x 514 or 512x512? We only need to update interior points

#data transfers: u0 to GPU, run simulation -> "return" u
@cuda.jit
def jacobi_kernel_2d(u,u_interior):
    i, j = cuda.grid(2)   #global grid
    
    # For kernel: remember bounce_check if suitable 
    if i > 1 and i < u.shape[0] - 2 and j > 1 and j < u.shape[1] - 2:
        if u_interior[i, j]:
            u[i, j] = 0.25 * (u[i, j - 1] + u[i, j + 1] + u[i - 1, j] + u[i + 1, j]) #Jacobi

def jacobi_helper(u, interior_mask, max_iter):
    d_u = cuda.to_device(u) # to GPU
    d_interior_mask = cuda.to_device(interior_mask)
    for _ in range(max_iter):
        jacobi_kernel_2d[bpg,tpb](d_u, d_interior_mask)
    cuda.synchronize() # ensure all kernels are done
    u = d_u.copy_to_host()
    return u

tpb = (32,32)  # thread per block 
bpg = (all_u0[0].shape[0] // tpb[0] + (all_u0[0].shape[0] % tpb[0] != 0), 
       all_u0[0].shape[1] // tpb[1] + (all_u0[0].shape[1] % tpb[1] != 0))  # block per grid

# compile the function
d_u = cuda.to_device(all_u0[0]) # to GPU
d_interior_mask = cuda.to_device(all_interior_mask[0])
jacobi_kernel_2d[bpg,tpb](d_u, d_interior_mask)
jacobi_kernel_2d[bpg,tpb](d_u, d_interior_mask)
jacobi_kernel_2d[bpg,tpb](d_u, d_interior_mask)
u = d_u.copy_to_host()

plt.imshow(u)
plt.colorbar()
plt.savefig('task8fig.png')
plt.close()

# args_list = [(u0, mask, MAX_ITER) for u0, mask in zip(all_u0, all_interior_mask)]

# exec_times = []
# all_u=np.empty_like(all_u0)

# for i, (u0,interior_mask, MAX_ITER) in enumerate(args_list):   
#     t=time()
#     u = jacobi_helper(u0,interior_mask,MAX_ITER)
#     t=time()-t 
#     all_u[i]=u
#     exec_times.append(t)
    
# #end result
# # print(building_ids)
# # print(exec_times)

# # SummaryDict = {}
# # #PrintsummarystatisticsinCSVformat
# # stat_keys=['mean_temp', 'std_temp','pct_above_18', 'pct_below_15']
# # print('Summary Statistics')
# # print('building_id, '+','.join(stat_keys)) #CSVheader
# # for bid,u, interior_mask in zip(building_ids,all_u, all_interior_mask):
# #     stats=summary_stats(u,interior_mask)
# #     # Create a dictionary to hold the statistics
# #     SummaryDict[bid] = stats
# #     # Print the building ID and summary statistics
# #     print(f"{bid},",",".join(str(round(stats[k],3)) for k in stat_keys))

# # # Create a DataFrame from the summary statistics dictionary
# # summary_df = pd.DataFrame.from_dict(SummaryDict, orient='index')
# # # Save the summary statistics to a CSV file
# # summary_df.to_csv('results/task8summary_statistics.csv', index_label='building_id')
    
# # #Save the results, in order to visualize them later
# # np.savez('results/task8_result.npz', exec_times = np.array(exec_times), 
# #         building_ids = np.array(building_ids), 
# #         all_u0 = np.array(all_u0),
# #         all_u = np.array(all_u),
# #         all_interior_mask = np.array(all_interior_mask))