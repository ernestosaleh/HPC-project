#!/bin/bash
#BSUB -J mini_proj_task_1to_3
#BSUB -q hpc
#BSUB -W 24:00
#BSUB -R "rusage[mem=2GB]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -R "span[hosts=1]"
#BSUB -n 4
#BSUB -o task_1to_3_%J.out
#BSUB -e task_1to_3_%J.err

# Initialize Python environment 
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

#Convenient message 
echo "Running simulation for 20 floorplans"

# Run Python script 
python task_1to_3.py 20
