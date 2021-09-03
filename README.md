# Keyword2Text

This repository contains the code of the paper: "Keyword2Text: A Plug-and-Play Method for Controlled Text Generation", if you find this useful and use it for your own research, please cite us.

## Setup

1. Download and unzip the repository.
2. Create a new conda environment and install the required libraries from the `requirements.txt` file.
```bash
conda create -n k2t python=3.6
conda activate k2t
pip install --file requirements.txt
```

A GPU will be required to run the experiments.
Make sure you have a results folder.



## Run Model

### Hyperparameter Study

Uncomment the appropriate lines of run.sh to run the hyperparameter experiments from the paper. For example, 

python main_DBS.py -mode='next' -file_name=/data/50_keywordsets_eval/word_sets.txt -results_subfolder=guide_vs_no_guide_beams -weight=10.0 -top_p=0.9 -n_generated_sentences=90 -do_guarantee=True

runs K2T with ordered guide words (mode='next') on the random keywords dataset. It runs with lambda=weight=10, nucleus sampling with top-p=0.9, number of generated tokens = 90, and no weight annealing to guarantee word appearance. The results are saved in results/tmp

### ROC Story dataset

Uncomment the appropriate line of run.sh to run the model on the ROC story dataset:

python main_DBS.py -mode='max' -file_name=/data/ROC/ROCStories_20_storylines_500_0.txt -results_subfolder=final4_ -weight=5.0 -top_p=0.9 -n_generated_sentences=-7 -n_beams=4 -do_guarantee=True

### News Article dataset

Uncomment the appropriate line of run.sh to run the model on the News Article story dataset:

python main_DBS.py -mode='max' -file_name=/data/keyword_to_articles -results_subfolder=tmp -key2article=True -weight=5.0 -top_p=0.9 -n_generated_sentences=-15 -n_beams=4 -do_guarantee=True


