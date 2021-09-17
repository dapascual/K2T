import numpy as np
import os
import pandas as pd
import scipy.io as sio
import math
import torch
import spacy
from transformers import GPT2Tokenizer, GPT2LMHeadModel
os.environ['GENSIM_DATA_DIR']='./gensim-data'

from nltk.stem import PorterStemmer, LancasterStemmer
porter = PorterStemmer()

from sklearn.metrics.pairwise import cosine_similarity
import itertools

from torch.utils.data import Dataset, DataLoader


def glove_encode(glove_encoder, word):
    return glove_encoder(word)



def checker(string):
    string = string.replace("'ve", '')
    string = string.replace("@", '')
    string = string.replace("'re", '')
    string = string.replace("'d", '')
    string = string.replace("?", '')
    string = string.replace("'s", '')
    string = string.replace(":", '')
    string = string.replace("!", '')
    string = string.replace('"', '')
    string = string.replace(".", '')
    string = string.replace("--", '')
    string = string.replace("'", '')
    string = string.replace(",", '')
    string = string.replace(';', '')
    string = string.replace('‘', '')
    string = string.replace('(', '')
    string = string.replace(')', '')
    string = string.replace('\'', '')
    string = string.replace(' ', '')
    return(string)


## Pytorch
def converter_table_glove():
    import gensim.downloader as api
    glove_encoder = api.load("glove-wiki-gigaword-300")

    path = str(os.path.dirname(os.path.abspath(__file__))) + \
        '/data/converter_table_glove'

    # load gpt-2 model
    #model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    #model.eval()

    holder = np.zeros((50257, 300))

    # translate every word from the gpt-2 space into a glove representation
    for i in range(50257):
        try:
            word = tokenizer.decode([i])
            word = checker(word.strip().lower())
            glove = glove_encoder[word]
            holder[i, :] = glove
        except:
            word = tokenizer.decode([i])
            holder[i, :] = np.zeros((300)) #+ 500

    # Save all 50'000 glove representations of the gpt-2 words
    np.save(file=path, arr=holder)
    print('Table was generated')
    
def converter_table_word2vec():
    import gensim.downloader as api
    word2vec_encoder = api.load("word2vec-google-news-300")

    path = str(os.path.dirname(os.path.abspath(__file__))) + \
        '/data/converter_table_word2vec'

    # load gpt-2 model
    #model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    #model.eval()

    holder = np.zeros((50257, 300))

    # translate every word from the gpt-2 space into a word2vec representation
    for i in range(50257):
        try:
            word = tokenizer.decode([i])
            word = checker(word.strip().lower())
            word2vec = word2vec_encoder[word]
            holder[i, :] = word2vec
        except:
            word = tokenizer.decode([i])
            holder[i, :] = np.zeros((300)) #+ 500

    # Save all 50'000 word2vec representations of the gpt-2 words
    np.save(file=path, arr=holder)
    print('Table was generated')


def count_word_stem_one(word, sequence):
    #print ("Sequence", sequence)
    sequence = sequence.split()

    word_count = 0
    word_stem = porter.stem(word.lower())
    for s_word in sequence:
        s_word_stem = porter.stem(s_word.lower())
        if(s_word_stem == word_stem):
            word_count = 1
            break

    return word_count

def count_word_stem(word, sequence):
    #print ("Sequence", sequence)
    sequence = sequence.split()
    word_count = 0

    word_stem = porter.stem(word.lower())

    for s_word in sequence:
        s_word_stem = porter.stem(s_word.lower())
        #print(s_word_stem)
        if(s_word_stem == word_stem):
            word_count += 1

    return word_count

# A score function for the quality of the sentence
def evaluate_quality(sequence, word, related_count, perplexity, guide, temp=1.):
    # we aim for one ocurance of the word,  and low perplexity
    w_1 = 1
    w_3 = 0.001 
    c_star = 2

    if(word == ""):
        quality_score = math.exp(-(w_1*(c_star) + w_3*perplexity))
        return quality_score

    quality_score = 0
    word_count = count_word_stem(word, sequence)
    

    if(word_count != 0) and guide:
        quality_score = math.exp(-(w_1*word_count + w_3*perplexity))
    else:
        quality_score = math.exp(-(w_1*(c_star) + w_3*perplexity))

    quality_score = quality_score/temp
    # DEBUG
    #print("txt, quality_score, word_count, rel_count, ppl", sequence, quality_score, word_count, related_count, perplexity)

    return quality_score, word_count



# A score function for the quality of the sentence
def evaluate_quality_linear(sequence, word_count, perplexity, temp=1., perp=False):
    # we aim for one ocurance of the word,  and low perplexity
    w_1 = 1
    w_3 = 0.01 

    if perp:
        quality_score = word_count - w_3*perplexity
    else:
        quality_score = word_count + w_3*perplexity

    quality_score = quality_score/temp # Temperature for sampling
   
    return quality_score


# simply the negative cosine similarity for use in calculating the 'best_tour'
def neg_cosine_similarity(v,w):
    return -1 * cosine_similarity(np.reshape(v, (1, -1)), np.reshape(w, (1, -1)))

# simply the positive cosine similarity for use in calculating the "worst" tour using 'best_tour' - used only as a sanity check (is worst worse than best?)
def pos_cosine_similarity(v,w):
    return cosine_similarity(np.reshape(v, (1, -1)), np.reshape(w, (1, -1)))

# function to calculate the total tour length when visiting the words in the given 'order'
def tour_length(distance_matrix, order):
    length = 0
    for i, j in zip(order, order[1:]):
        length += distance_matrix[i][j]
    return length

# find the best tour through the guide words, minimizing the pairwise distance between consecutive guide words
def best_tour(glove_array, distance=neg_cosine_similarity, top_k=1):
    """
    returns the best order to minimize the tour length
    default pairwise distance is the negative cosine similarity
    input should be an nxm array of word embeddings, where n is no. words and m is length of the word embedding
    *NOT IMPLEMENTED: set top_k to the beam length if you want to use a separate order per beam.
    """
    n = len(glove_array)
    distance_matrix = np.zeros((n,n))
    for i, v in enumerate(glove_array):
        for j, w in enumerate(glove_array):
            distance_matrix[i][j] = distance(v,w)
    tours = {}
    for i in itertools.permutations(list(range(n))):
        tours[i] = tour_length(distance_matrix, i)
        best_tour = min(tours, key=tours.get)
    return best_tour


class KeywordsDataset(Dataset):
    """Keywords dataset."""

    def __init__(self, keyword_sets):
        self.keyword_sets = keyword_sets

    def __len__(self):
        return len(self.keyword_sets)

    def __getitem__(self, idx):
        return self.keyword_sets[idx]
