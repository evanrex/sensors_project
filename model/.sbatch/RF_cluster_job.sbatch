#!/bin/bash
#SBATCH --job-name=riverRF
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --partition=stampede
#SBATCH --output=/home-mscluster/erex/sensors_project/model/experiments/.out/result-RF-%A-%a.out
#SBATCH --array=1-10 # job array index

echo "running job: "${SLURM_ARRAY_TASK_ID}
# mkdir -p /home-mscluster/erex/sensors_project/model/experiments/saving_dir${SLURM_ARRAY_TASK_ID}


python -m main \
--data_path /home-mscluster/erex/sensors_project/data/river_sensor_data.csv \
--saving_dir /home-mscluster/erex/sensors_project/model/experiments/saving_dir${SLURM_ARRAY_TASK_ID} \
--random_state ${SLURM_ARRAY_TASK_ID} \
--architecture RF

