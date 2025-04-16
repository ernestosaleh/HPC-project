#!/bin/bash
#BSUB -J check_cpu
#BSUB -q hpc
#BSUB -W 24:00
#BSUB -R "rusage[mem=1GB]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -R "span[hosts=1]"
#BSUB -n 4
#BSUB -o check_cpu%J.out
#BSUB -e check_cpu%J.err

lscpu > lscpu_output.txt
echo "lscpu output saved to lscpu_output.txt"
echo "Checking CPU model..."
