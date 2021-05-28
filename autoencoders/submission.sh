#!/bin/bash
 
#SBATCH --job-name constel 
#SBATCH --mem 32000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH -t 0:20:00

module purge
module load tensorflow2/py3.cuda10.0

#python autoencoderCluster.py  > log_ae.out
python Denoise-VAE-keras.py > log_vae2.out
