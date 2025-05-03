#!/bin/bash
#BSUB -J mini_proj_task_8
#BSUB -q c02613
#BSUB -W 1:00
#BSUB -R "rusage[mem=3GB]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -o bash_output/task8_%J.out
#BSUB -e bash_output/task8_%J.err

# Initialize Python environment 
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# Run Python script 
python task8.py 20
