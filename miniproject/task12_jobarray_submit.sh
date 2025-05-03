#!/bin/bash
#BSUB -J mini_proj_task_12[1-10]
#BSUB -q hpc
#BSUB -W 24:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -R "span[hosts=1]"
#BSUB -n 4
#BSUB -o bash_output/task12_jobarray_%J_%I.out
#BSUB -e bash_output/task12_jobarray_%J_%I.err

# Initialize Python environment 
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# Run the Numba JIT implementation using jobarrays
python task12_jobarrays.py 20 4 10 $LSB_JOBINDEX #Expects nr floorplans, nr of processors and then job_index
