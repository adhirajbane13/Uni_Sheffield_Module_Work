#!/bin/bash
#SBATCH --job-name=Liability_Claim_Prediction
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4


module load Java/17.0.4
module load Anaconda3/2022.05
source activate myspark


spark-submit --master local[*] ../Q2_code.py