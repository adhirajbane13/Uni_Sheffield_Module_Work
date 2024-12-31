#!/bin/bash
#SBATCH --job-name=nasa_log_analysis
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4


module load Java/17.0.4
module load Anaconda3/2022.05
source activate myspark


spark-submit --master local[*] ../Q1_code.py
