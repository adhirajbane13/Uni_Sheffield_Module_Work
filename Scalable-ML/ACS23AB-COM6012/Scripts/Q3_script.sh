#!/bin/bash
#SBATCH --job-name=Higgs_Boson_Detection
#SBATCH --time=06:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4


module load Java/17.0.4
module load Anaconda3/2022.05
source activate myspark


spark-submit --master local[*] ../Q3_code.py
