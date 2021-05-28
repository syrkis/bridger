#!/bin/bash

#SBATCH --job-name=zipper            # Job name
#SBATCH --output=%x.out      # Name of output file (%j expands to jobId)
#SBATCH --cpus-per-task=1        # Schedule one core
#SBATCH --time=01:00:00          # Run time (hh:mm:ss) - run for one hour max
# #SBATCH --gres=gpu
#SBATCH --partition=brown    # Run on either the Red or Brown queue
# #SBATCH --mail-type=FAIL,END          # Send an email when the job finishes or fails
#SBATCH --mem=20000             # memo

zip -r /home/timp/npys.zip /home/timp/repositories/bringo/data/npys/* 

