import time
import torch
import json
import os
import numpy as np
import scipy.io as sio
import argparse
os.environ['TRANSFORMERS_CACHE']='./transformer_models'  # Work-around to avoid memory problems in server, comment out depending on memory availability
from utility_gpt import *
from perplexity import *
from collections import Counter
from scipy import stats
import operator

from transformers import GPT2Tokenizer

#Self Bleu
import random
from functools import partial
from multiprocessing.pool import Pool

import spacy
from tqdm import tqdm
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("-file", type=str)
    parser.add_argument("-zipf_N", type=int, default=5000)
    parser.add_argument("-dist_N", type=int, default=2, help="N in distinct-N metric")
    parser.add_argument("--n_sample", type=int, default=50,
                        help="how many sentences to sample to calculate bleu")

    return parser.parse_args()

def distinct_n(example, n, n_distinct, n_total, counter):
    #counter = Counter()
    #n_total = 0
    #n_distinct = 0
    #for example in examples:
    for token in zip(*(example[i:] for i in range(n))):
        if token not in counter:
            n_distinct += 1
        elif counter[token] == 1:
            n_distinct -= 1
        counter[token] += 1
        n_total += 1
    return n_distinct, n_total, counter

def bleu_i(weights, all_sentences, smoothing_function, i):
    # noinspection PyTypeChecker
    return sentence_bleu(
        references=all_sentences[:i] + all_sentences[i + 1:],
        hypothesis=all_sentences[i],
        weights=weights,
        smoothing_function=smoothing_function)

def main():

    args = parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')

    # Self-BLEU
    random.seed(0)
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])
    nlp.add_pipe(nlp.create_pipe('sentencizer'))

    all_sentences = []
    #save_path = str(os.path.dirname(os.path.abspath(__file__))) + "/results/50_keywordsets_eval/progressive/"
    save_path = os.path.splitext(args.file)[0] +"_metrics"
    

    with open(args.file, 'r') as file1:  
        lines = file1.readlines()
        ppl = np.zeros((50))
        i = 0
        text = ''
        next_yes = False
        next_no = False
        sep = '<|endoftext|>'
        counter_2 = Counter()
        n_total_2 = 0
        n_distinct_2 = 0
        counter_3 = Counter()
        n_total_3 = 0
        n_distinct_3 = 0
        counter_4 = Counter()
        n_total_4 = 0
        n_distinct_4 = 0
        cnt_zipf = Counter()

        for line in tqdm(lines):    
            if "Final sequence:" in line:
                next_no = True
            elif next_no:
                next_yes = True
                next_no = False
                
            if "Success_rate:" in line:
                next_yes = False        
                stripped = text.split(sep, 2)[1]
                #### Put metrics here!:            
                tokenized_text = tokenizer.encode(stripped)
                # Distinct n:
                n_distinct_2, n_total_2, counter_2 = distinct_n(tokenized_text, 2, n_distinct_2, n_total_2, counter_2) 
                n_distinct_3, n_total_3, counter_3 = distinct_n(tokenized_text, 3, n_distinct_3, n_total_3, counter_3) 
                n_distinct_4, n_total_4, counter_4 = distinct_n(tokenized_text, 4, n_distinct_4, n_total_4, counter_4) 
                # Zipf coeff:
                cnt_zipf.update(tokenized_text)
                # Self-BLEU:
                all_sentences.append(tokenized_text)
                # PPL: 
                ppl[i] = distilGPT2_perplexity_score(stripped)
                i = i+1
                text = ''
            elif next_yes:
                text = text + line

    text_file = open(save_path + '.txt', 'a+', encoding='utf8')
    print(f"{os.path.basename(args.file)}")
    text_file.write(f"{os.path.basename(args.file)}\n")
    # Perplexity
    mean_ppl = np.mean(ppl)
    print(f"perplexity\t {mean_ppl}")
    text_file.write(f"perplexity\t {mean_ppl}\n\n")
    
    # Distinct n 2
    print(f"distinct 2-grams\ttotal 2-grams\tdistinct proportion")
    text_file.write(f"distinct 2-grams\ttotal 2-grams\tdistinct proportion\n")
    print(f"{n_distinct_2}\t{n_total_2}\t{n_distinct_2/n_total_2}")
    text_file.write(f"{n_distinct_2}\t{n_total_2}\t{n_distinct_2/n_total_2}\n\n")
    
    # Distinct n 3
    print(f"distinct 3-grams\ttotal 3-grams\tdistinct proportion")
    text_file.write(f"distinct 3-grams\ttotal 3-grams\tdistinct proportion\n")
    print(f"{n_distinct_3}\t{n_total_3}\t{n_distinct_3/n_total_3}")
    text_file.write(f"{n_distinct_3}\t{n_total_3}\t{n_distinct_3/n_total_3}\n\n")
    
    # Distinct n 4
    print(f"distinct 4-grams\ttotal 4-grams\tdistinct proportion")
    text_file.write(f"distinct 4-grams\ttotal 4-grams\tdistinct proportion\n")
    print(f"{n_distinct_4}\t{n_total_4}\t{n_distinct_4/n_total_4}")
    text_file.write(f"{n_distinct_4}\t{n_total_4}\t{n_distinct_4/n_total_4}\n\n")
    
    # Zipf coeff
    xs = np.arange(1, min(len(cnt_zipf), args.zipf_N)+1)
    ys = np.array(sorted(cnt_zipf.values(), key=operator.neg)[:args.zipf_N])
    a, b, r, p, std = stats.linregress(np.log(xs), np.log(ys))
    print("zipf value\tregression r value \tregression p value")
    text_file.write("zipf value\tregression r value \tregression p value\n")
    print(f"{-a}\t{-r}\t{p}")
    text_file.write(f"{-a}\t{-r}\t{p}\n\n")
    """
    #############
    # Self-BLUE:
    smoothing_function = SmoothingFunction().method1
    pool = Pool(processes=os.cpu_count())
    bleu_scores = []
    for n_gram in range(1, 6):

        if n_gram == 1:
            weights = (1.0, 0, 0, 0)
        elif n_gram == 2:
            weights = (0.5, 0.5, 0, 0)
        elif n_gram == 3:
            weights = (1.0 / 3, 1.0 / 3, 1.0 / 3, 0)
        elif n_gram == 4:
            weights = (0.25, 0.25, 0.25, 0.25)
        elif n_gram == 5:
            weights = (0.2, 0.2, 0.2, 0.2, 0.2)
        else:
            raise ValueError
        #print("Len all sentences: ", len(all_sentences))
        bleu_scores.append(
            list(tqdm(
                pool.imap_unordered(
                    partial(bleu_i, weights, all_sentences, smoothing_function),
                    random.sample(range(len(all_sentences)), args.n_sample)),
                total=args.n_sample,
                smoothing=0.0,
                desc=f"bleu-{n_gram}")))
        print(f"\n\nbleu-{n_gram} = {sum(bleu_scores[n_gram - 1]) / args.n_sample}")
        text_file.write(f"\n\nbleu-{n_gram} = {sum(bleu_scores[n_gram - 1]) / args.n_sample}\n")

    for n_gram in range(5):
        print(f"bleu-{n_gram + 1} = {sum(bleu_scores[n_gram]) / args.n_sample}")
        text_file.write(f"bleu-{n_gram + 1} = {sum(bleu_scores[n_gram]) / args.n_sample}\n")
    ################
    """
    save_dict = {}
    save_dict['ppl'] = mean_ppl
    save_dict['2_distinct'] = [2, n_distinct_2, n_total_2, n_distinct_2/n_total_2]
    save_dict['3_distinct'] = [3, n_distinct_3, n_total_3, n_distinct_3/n_total_3]
    save_dict['4_distinct'] = [4, n_distinct_4, n_total_4, n_distinct_4/n_total_4]
    save_dict['zipf'] = [a, r, p]
    """
    save_dict['self_bleu'] = [sum(bleu_scores[0]) / args.n_sample,   \
                                sum(bleu_scores[1]) / args.n_sample, \
                                sum(bleu_scores[2]) / args.n_sample, \
                                sum(bleu_scores[3]) / args.n_sample, \
                                sum(bleu_scores[4]) / args.n_sample]
    """
    np.save(save_path + ".npy", save_dict)


if __name__ == '__main__':
    main()

