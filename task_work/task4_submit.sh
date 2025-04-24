#!/bin/bash
#BSUB -J mini_proj_task_4
#BSUB -q hpc
#BSUB -W 24:00
#BSUB -R "rusage[mem=1GB]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -R "span[hosts=1]"
#BSUB -n 4
#BSUB -o task4_%J.out
#BSUB -e task4_%J.err

# Initialize Python environment 
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613


#Convenient message 
echo "Profiling reference implementation of jacobi"

# Run Python script 
kernprof -l task4.py 1

#Output the result from profiling
python -m line_profiler -rmt task4.py.lprof

#remove .lprof file
rm task4.py.lprof
