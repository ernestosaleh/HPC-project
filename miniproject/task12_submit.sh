#!/bin/bash
#BSUB -J mini_proj_task_12
#BSUB -q hpc
#BSUB -W 24:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -R "span[hosts=1]"
#BSUB -n 25
#BSUB -o bash_output/task12_%J.out
#BSUB -e bash_output/task12_%J.err

# Initialize Python environment 
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# Run the Numba JIT implementation for all floor plans with 25 processors
python task12.py 4571 25
