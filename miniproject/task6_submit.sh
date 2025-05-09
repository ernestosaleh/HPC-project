#!/bin/bash
#BSUB -J mini_proj_task_6
#BSUB -q hpc
#BSUB -W 24:00
#BSUB -R "rusage[mem=1GB]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -R "span[hosts=1]"
#BSUB -n 32
#BSUB -o bash_output/task6v2_%J.out
#BSUB -e bash_output/task6v2_%J.err

# Initialize Python environment 
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# Run Python script 
python task6.py 100 32
