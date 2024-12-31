#!/bin/bash
#SBATCH --job-name=Movie_Lens_Recommendation
#SBATCH --time=01:00:00
#SBATCH --mem=35G
#SBATCH --cpus-per-task=32


module load Java/17.0.4
module load Anaconda3/2022.05
source activate myspark


spark-submit --master local[*] \
             --conf "spark.executor.memory=30g" \
             --conf "spark.driver.memory=30g" \
             ../Q4_code.py
