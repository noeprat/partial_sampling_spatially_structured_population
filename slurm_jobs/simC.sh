#!/bin/bash
#SBATCH -J simC                        # name of the job
#SBATCH -o ./out_simC%A_%a                     # name of the output file (%A_%a ensures you have a different name for each job: %A is th job number, %a the array number)
#SBATCH -e ./out_simC%A_%a                     # name of error file (same as output file -> all in one file)
#SBATCH -D /home/prat/                                     # working directory
#SBATCH --array=1-5                                      # this is an array job, where you launch one job per array item (and you can pass different parameters to each job, see below)
#SBATCH --ntasks 1 
#SBATCH --cpus-per-task 1                                  # number of CPUs for each job
#SBATCH --time 1-00:00:00                                    # maximum allowed time (3 days)
#SBATCH --qos=serial                                       # must add to avoid problems with billing
#SBATCH --mail-type=BEGIN,FAIL,END                         # receive an email if job begins, fail, or ends
#SBATCH --mail-user=noe.prat@epfl.ch                  # change to your email 

module purge
module load  gcc/11.3.0
module load intel/2021.6.0
module load python/3.10.4

source venvs/noe-sim-venv/bin/activate

N_graph='100'
N_wm='1000'
M_graph='50'
M_wm='500'
log_s_min='-4'
log_s_max='-1'
## nb_trajectories='100' ## 100 runs
nb_trajectories='1000000'   ## 1M runs
migration_rate=`head -n $SLURM_ARRAY_TASK_ID parameters_simC.txt | tail -n 1 | cut -d' ' -f1`  ## to sweep m
nb_demes='10'


python3 /home/prat/slurm-main.py clique $SLURM_ARRAY_TASK_ID $N_graph $M_graph $log_s_min $log_s_max $nb_trajectories $migration_rate $nb_demes



## sbatch simC.sh