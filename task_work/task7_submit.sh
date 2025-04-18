#!/bin/bash
#BSUB -J mini_proj_task_7
#BSUB -q hpc
#BSUB -W 24:00
#BSUB -R "rusage[mem=3GB]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -R "span[hosts=1]"
#BSUB -n 4
#BSUB -o task7_%J.out
#BSUB -e task7_%J.err

# Initialize Python environment 
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# Run Python script 
python task7.py 30
