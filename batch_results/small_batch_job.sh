#!/bin/bash
#SBATCH -A research
#SBATCH -n 40 
#SBATCH --partition=long
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2GB
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH --output=op_file.txt

#module load cuda/10.1

source /home2/shravya.k/venv/bin/activate

#bash /home2/shravya.k/ShannonAI-mrc-for-flat-nested-ner/script/train_en_genia.sh
mkdir -p /scratch/shravya.k/
touch /scratch/shravya.k/testing

PATH_SHARE=${SLURM_SUBMIT_DIR}/../../../../share1/shravya.k/ 
cp /scratch/shravya.k/testing ${PATH_SHARE}/testing 
