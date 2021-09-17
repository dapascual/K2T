import time
import torch
import json
import os
import numpy as np
import scipy.io as sio
import argparse


import gensim.downloader as api
import pickle
import argparse


######## Change to encode keywords for the desired task
encode_50keywords = True
encode_ROC = False
encode_articles = False

print('word2vec loading...')
word2vec_encoder = api.load("word2vec-google-news-300")
print('word2vec loaded')
word2vec_dict = {}

######## Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-file', type=str)
args = parser.parse_args()
file_name = args.file
folder = os.path.dirname(file_name)

print('file_name: ', file_name)

if encode_50keywords == True:
    
    file1 = open(str(os.path.dirname(os.path.abspath(__file__))) +
                 file_name, "r+")
    lines = file1.readlines()

    i=0
    for line in lines:
        keywords = list(line.strip().split(", "))
        word2vec_words = []
        print(keywords)
        for word in keywords:
            word2vec = word2vec_encoder[word]
            word2vec_words.append(word2vec)
            word2vec_dict[word] = word2vec
        save_path = str(os.path.dirname(
            os.path.abspath(__file__))) + folder + '/set_' +str(i) + '.npy'

        np.save(save_path, word2vec_words)
        i=i+1

    save_path_dict = str(os.path.dirname(
        os.path.abspath(__file__))) + folder + '/dict_word2vec.pkl'
    with open(save_path_dict, 'wb') as f:
        pickle.dump(word2vec_dict, f, pickle.HIGHEST_PROTOCOL)

