#!/bin/bash
#SBATCH --job-name=mdacc_sclc
#SBATCH --output=logs/batch160_%j.out
#SBATCH --error=logs/batch160_%j.err
#SBATCH --partition=256GB
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G	
#SBATCH --time=48:00:00

nohup python -u batch_process_feature.py > logs/SCLC_MDACC2.log 2>&1 &
