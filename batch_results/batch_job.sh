#!/bin/bash
#SBATCH -A dma
#SBATCH -n 40 
#SBATCH --partition=long
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2GB
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH --output=op_file.txt

#module load cuda/10.1

source /home2/shravya.k/venv/bin/activate

bash /home2/shravya.k/SEBI-MRC-NER/script/train_en_genia.sh

#cp -r /scratch/shravya.k/mrc-ner /share1/shravya.k/