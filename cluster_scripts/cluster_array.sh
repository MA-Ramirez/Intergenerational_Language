#!/bin/bash

#SBATCH --job-name=MAR 		                 #Name of the job
#SBATCH --array=1-500%100                     #500 tasks (100 are concurrent)
#SBATCH --nodes=1				              #Each task lives on one node
#SBATCH --ntasks-per-node=1				      #One Julia process per task
#SBATCH --cpus-per-task=1		              #Serial inner sweep to ensure isolation
#SBATCH --mem=4000	                        #Memory in Mb per CPU (TUNE after profiling)
#SBATCH --time=04:00:00                    #Walltime ceiling (TUNE after profiling)
#SBATCH --mail-user=ramirez@mis.mpg.de
#SBATCH --mail-type=FAIL                  #Send email at BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=logs/slurm-%A_%a.out      #Separate error output per run, not only one for the whole array
#SBATCH --error=logs/slurm-%A_%a.err

module load julia/1.11.4
  julia --project=. scripts/RunScript.jl $SLURM_ARRAY_TASK_ID
module unload julia/1.11.4