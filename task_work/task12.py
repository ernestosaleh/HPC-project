from code_base import load_data, jacobi, summary_stats
import numpy as np
import pandas as pd
import sys

LOAD_DIR= '/../../dtu/projects/02613_2025/data/modified_swiss_dwellings/'

with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
    building_ids=f.read().splitlines()
    
    
 #Loadfloorplans, maybe not load all at once
all_u0=np.empty((N, 514,514))
all_interior_mask=np.empty((N,512,512),dtype='bool')
for i,bid in enumerate(building_ids):
    u0,interior_mask =load_data(LOAD_DIR,bid)
    all_u0[i]=u0
    all_interior_mask[i]=interior_mask

######################################################################
# A very fast implementation of the simulation should be included here
######################################################################

######################################################################
# END OF SIMULATION
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
summary_df.to_csv('results/task1_to_3summary_statistics.csv', index_label='building_id')