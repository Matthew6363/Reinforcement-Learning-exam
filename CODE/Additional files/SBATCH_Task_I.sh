#!/bin/bash
#SBATCH --job-name=RL-Task                    # job name
#SBATCH --output=../LOGs/log_RL_%j.txt        # print placement
#SBATCH --error=../LOGs/log_RL_error_%j.txt   # error log
#SBATCH --partition=EPYC                      # Partizione GPU
#SBATCH --nodes=1                             # 1 Nodo
#SBATCH --cpus-per-task=4                     # 4 CPU (per il num_workers=4)
#SBATCH --mem=32G                             # Memoria RAM
#SBATCH --time=00:30:00                       # Tempo limite



echo "Job ID: $SLURM_JOB_ID"
echo "Processing Batch: $BATCH_NUM"

python3 -m venv ../QSGRM_env
source ../QSGRM_env/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

# module load python/3.10
cd CODE
python -u  simulation_Task_I.py
