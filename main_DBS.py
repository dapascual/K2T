import time
import torch
import json
import os
import numpy as np
import scipy.io as sio
import argparse
import gc

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn.functional as F

from sklearn.metrics.pairwise import cosine_similarity
from utility_gpt import *
from perplexity import *
import pickle
import random
from encode_keywords import create_enc_dict
from collections import Counter

from nltk.stem import PorterStemmer, LancasterStemmer
porter = PorterStemmer()


word_embedding = {
    'glove': "glove-wiki-gigaword-300",
    'word2vec': "word2vec-google-news-300"
    }


if not os.path.exists(str(os.path.dirname(os.path.abspath(__file__))) + '/data/converter_table_glove.npy'):
    print("Generating table of cosine distances...")
    converter_table_glove()

if not os.path.exists(str(os.path.dirname(os.path.abspath(__file__))) + '/data/converter_table_word2vec.npy'):
    print("Generating table of cosine distances...")
    converter_table_word2vec()


def distinct_n(example, n, n_distinct, n_total, counter):
    """
    Gives the number of distinct n-grams as well as the total n-grams
    Args:
        example: input text
        n: n-grams size (i.e., the n)
        n_distinct: distinct n-grams in previous iteration
        n_total: total n-grams in previous iteration
        counter: token counter in previous iteration, i.e., how many times a token appeared
        
    """
    for token in zip(*(example[i:] for i in range(n))):
        if token not in counter:
            n_distinct += 1
        elif counter[token] == 1:
            n_distinct -= 1
        counter[token] += 1
        n_total += 1
    return n_distinct, n_total, counter


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        #indices_to_remove = torch.zeros_like(logits, dtype=torch.uint8).scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove )

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        
    return logits
    
def get_keywords(keywords, enc_dict, tokenizer, mode):
    keywords_ = [w for w in keywords]

    # Select the next guide word(s)
    if keywords_:
        if mode=='next':
            keywords_ = [keywords_[0]]
        if mode=='random':
            keywords_ = [random.choice(keywords_)]
    
    keywords_enc = [enc_dict[w] for w in keywords_]
    keywords_gpt = {tokenizer.encode(w)[0]:w for w in keywords_}
    
    return keywords_enc, keywords_gpt
    
def get_logits(model, tokenizer, text, this_sequence, temperature):
    ## GPT2 - generate logits
    indexed_tokens = tokenizer.encode(text)
    indexed_this_seq = tokenizer.encode(this_sequence)
    tokens_tensor = torch.tensor([indexed_tokens])
    tokens_tensor = tokens_tensor.to('cuda')
    
    # Predict all tokens
    outputs = model(tokens_tensor)
    
    del tokens_tensor
    torch.cuda.empty_cache()
    
    logits = outputs.logits
    logits = logits[0, -1, :]/ temperature
    
    return logits, indexed_tokens, indexed_this_seq
    
def get_sim(keywords_enc, keywords_gpt, converter_table, guarantee, mode, only_max):

    if len(keywords_enc)>1:
        sims = np.array([cosine_similarity(np.reshape(w, (1, -1)), converter_table) for w in keywords_enc])
        if guarantee:
            for i, w in enumerate(keywords_gpt):
                sims[i][0][w] = 1
        if mode=='max':
            sim = np.max(sims, axis=0)
        elif mode=='all':
            sim = np.sum(sims, axis=0)
        else:
            raise Exception("keywords_enc length is greater than 1 so expect to be in mode 'max' or 'all'")
    else:
        sim = cosine_similarity(np.reshape(keywords_enc[0], (1, -1)), converter_table)
        
    # Only the target word, not the neighbour (as measured by cosine similarity)
    if only_max == True:
        sim_aux = np.zeros_like(sim)
        sim_aux[0,sim.argmax()] = sim.max()
        sim = np.squeeze(sim_aux)
    else:
        sim = np.clip(np.squeeze(sim), a_min=0, a_max=None) #tf.square(sim)  

    return sim
    
def get_weight(weight, guarantee, T_time, time):

    if guarantee:
        if T_time == 0:
            T_time = 1
        rate = (1/T_time)*np.log(100/weight)  # 100 is the maximum value the weight will reach
        weight = weight*np.exp(rate*time)

    return weight

def get_prediction(tokenizer, indexed_tokens, indexed_this_seq, keywords_gpt, predicted_index, guarantee, T_time, time):

    if guarantee and time > T_time:
        predicted_index = list(keywords_gpt.keys())[0]   
    if guarantee and predicted_index in keywords_gpt:
        predicted_text = tokenizer.decode(indexed_tokens) + ' ' + keywords_gpt[predicted_index]
        this_sequence = tokenizer.decode(indexed_this_seq) + ' ' + keywords_gpt[predicted_index]
        pred_word = keywords_gpt[predicted_index]
    else:
        predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
        this_sequence = tokenizer.decode(indexed_this_seq + [predicted_index])
        pred_word = predicted_text.split()[-1].split('<|endoftext|>')[-1]
        
    return pred_word, predicted_text, predicted_index, this_sequence



def sample_sentence(text, this_sequence, tokenizer, model, keywords, enc_dict, guide_probs, converter_table, weight, guide=False, prev_proba=1, top_k=0, top_p=0.9, temperature=1., only_max=False, mode='max', guarantee=False, time=0, T_time=1, det_BS=False, ith=0):
    """ Samples the next word of the sequence with logit modification (guidance)
    Modes:
        mode='max':     each token is shifted by the cosine similarity to the closest guide word
        mode='all':     each token is shifted by the cosine similarity to each guide word
        mode='next':    the order of the guide words is fixed and each token is shifted towards the next guide word in the sequence
        mode='random':  a random word is selected from the remaining (not yet appeared) guide words and each token is shifted towards this guide word
    """    
    
    # Get word stems, encode keywords and get logits from LM from context
    guide_word_stems = [porter.stem(w.lower()) for w in keywords]    
    keywords_enc, keywords_gpt = get_keywords(keywords, enc_dict, tokenizer, mode)
    logits, indexed_tokens, indexed_this_seq = get_logits(model, tokenizer, text, this_sequence, temperature)
    
    # Get probabilities for ppl calculation and log-softmax of logits for modification
    proba = F.softmax(logits, dim=-1) 
    logits = F.log_softmax(logits, dim=-1)  
    
    # Calculate cosine similarity, weight with annealing and modify logits
    if keywords_enc and guide:
        sim = get_sim(keywords_enc, keywords_gpt, converter_table, guarantee, mode, only_max)        
        weight = get_weight(weight, guarantee, T_time, time)
        logits = logits + torch.tensor(sim*weight).cuda() #

    ## Sample tokens
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p) ###  
    logits = F.softmax(logits, dim=-1)          
    # Deterministic beam search or sampling, if p!=1. it means nucleus sampling
    
    predicted_index = 50256
    while guide and predicted_index == 50256:
        if det_BS:
            predicted_index = torch.topk(logits, ith+1)[1][ith].item()
        else:
            predicted_index = torch.multinomial(logits, 1).item()

    # Get predicted word and indices
    pred_word, predicted_text, predicted_index, this_sequence = get_prediction(tokenizer, indexed_tokens, indexed_this_seq, keywords_gpt, predicted_index, guarantee, T_time, time)
        
    # Update counters if word was predicted
    pred_word_stem = porter.stem(pred_word.lower())
    guide_next = guide
    time_next = time+1
    T_time_next = T_time
    if pred_word_stem in guide_word_stems:
        ind = guide_word_stems.index(pred_word_stem)
        keywords = keywords[:ind] + keywords[ind+1:]
        guide_probs = guide_probs + [(pred_word_stem, proba[predicted_index].item())]
        guide_next = False
        time_next = 1
        T_time_next = T_time-time+1
        
    return predicted_text, keywords, guide_next, guide_probs, prev_proba*proba[predicted_index], this_sequence, time_next, T_time_next

   

def sample_sentence_noguide(text, this_sequence, tokenizer, model, prev_proba=1, top_k=0, top_p=0.9, temperature=1., eos_c=0, det_BS=False, ith=0):
    """ Samples the next word of the sequence without logit modification (guidance)
    """   
    logits, indexed_tokens, indexed_this_seq  = get_logits(model, tokenizer, text, this_sequence, temperature)
    
    proba = F.softmax(logits, dim=-1)        
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    logits = F.softmax(logits, dim=-1) 

    if det_BS:
        predicted_index = torch.topk(logits, ith+1)[1][ith].item()
    else:
        predicted_index = torch.multinomial(logits, 1).item()
    
    predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
    this_sequence = tokenizer.decode(indexed_this_seq + [predicted_index])
    pred_word = predicted_text.split()[-1]  

    if predicted_index == 50256:
        eos_c_next=1
    else:
        eos_c_next=eos_c
    
    if eos_c == 1:
        next_proba = prev_proba
    else:
        next_proba = prev_proba*proba[predicted_index].item()

    return predicted_text, next_proba, this_sequence, eos_c_next

def get_score(k, number_of_words_per_sentence, online_probability, proba):
    alpha = 0.6
    length = (k+1)*number_of_words_per_sentence                 
    len_norm = ((5+length)**alpha)/(6**alpha)
    score_ = np.log(online_probability*proba)/len_norm

    return score_
    

def get_success_length(success_length, number_of_beams, guide, number_of_generated_sentences):
    
    for b in range(number_of_beams):    
        if guide[b]:
            success_length[b] = 0
    
    success_length = success_length[0]/number_of_generated_sentences
    return success_length
    
def get_success_rate(number_keywords, count_word_stem, keywords, full_text):
    # Success rate
    target_words = number_keywords
    target_count = 0
    for i in range(number_keywords):
        if count_word_stem(keywords[i], full_text[0]) > 0:
            target_count += 1
            
    success_rate = word_c[0]/number_keywords #target_count/target_words    
    
def conditional_language_generation(
    model,
    tokenizer,
    keyword_set,
    model_name='gpt2-large',
    enc_dict={},
    seed=None,
    temperature=1.,
    top_k=0,
    top_p=0.9,
    constant=20,
    number_of_concurrent_sentences = 10,
    number_of_generated_sentences = 20,
    number_of_words_per_sentence = 5,
    number_of_beams = 3,
    save_path='dummy.txt',
    only_max = False,
    no_do_wc=False,
    mode='max',
    do_guarantee=False,
    embedding='glove',
    det_BS=False,
    folder_name='',
    guide=True
):
    """
    Main function for conditional language generation
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    :top_p=1 Top_p is the cummulative probability used for nucleus sampling. 1 means no nucleus sampling
    :constant: How much are anchors weighted
    :counter index of wordset which is currently evaluated
    :TODO ns.....
    :modes:
        mode='max':     each token is shifted by the cosine similarity to the closest guide word
        mode='all':     each token is shifted by the cosine similarity to each guide word
        mode='next':    the order of the guide words is fixed and each token is shifted towards the next guide word in the sequence
        mode='random':  a random word is selected from the remaining (not yet appeared) guide words and each token is shifted towards this guide word
        mode='best_tour': 
        mode='worst_tour':
    """
    
    start_time = time.time()
    total_words = number_of_words_per_sentence*number_of_generated_sentences
    
###################################
	## Load words
    
    # Define task, keyword to article, keyword to story (ROC) or keyword to phrase
    in_text, keywords = keyword_set    
    keywords_enc = [enc_dict[w] for w in keywords]
    number_keywords = len(keywords)
    print(in_text, keywords)   
    print("N keywords: ", number_keywords)
     
    if mode=='best_tour':
        best_order = best_tour(keywords_enc)
        keywords_enc = [keywords_enc[i] for i in list(best_order)]
        print("Keywords: ", keywords, best_order)
        keywords = [keywords[i] for i in best_order]
        print("Keywords: ", keywords)
        mode = 'next' # Switch over to next (ordered) mode with the optimized order
       
###################################

    # File to save results as .txt
    text_file = open(save_path + '.txt', 'a+', encoding='utf8')
    text_file_sentences = open(save_path + 'SENTENCES.txt', 'a+', encoding='utf8')
       
    # prepare variables...
    #np.random.seed(seed)
    weight = constant
    converter_table = np.load(str(os.path.dirname(
        os.path.abspath(__file__))) + '/data/converter_table_' + str(embedding) + '.npy')  
    
    full_text = [in_text] * number_of_beams
    guide_words_s = [keywords]*number_of_beams
    guide_probs_s = [[]]*number_of_beams
    cum_quality_score = [0]*number_of_beams
    word_c = [0]*number_of_beams
    success_length =  [0]*number_of_beams
    online_probability = [1]*number_of_beams
    guide = [guide]*number_of_beams
    eos_count = [0]*number_of_beams
    total_time = [total_words-number_keywords]*number_of_beams
    current_time = [1]*number_of_beams    
    
    for k in range(number_of_generated_sentences):  
        # Define guidance word and index for guidance in model and quality function     
        result_subsequences = []
        for b in range(number_of_beams):
####################################### Generation loop ################################################
            for i in range(number_of_concurrent_sentences):                
                # Reset variables:
                context = full_text[b]
                guide_words = guide_words_s[b]
                guide_probs = guide_probs_s[b]
                # print(guide_probs)
                proba = 1
                this_sequence = ""
                w_c = 0
                eos_c = eos_count[b]
                t_time = total_time[b]
                c_time = current_time[b]
                if guide[b]:
                    guide_next = True    
                    for j in range(number_of_words_per_sentence):                        
                        context, guide_words, guide_next, guide_probs, proba, this_sequence, c_time, t_time = sample_sentence(context, 
                                this_sequence, tokenizer, model, guide_words, enc_dict, guide_probs, converter_table, 
                                weight, guide_next, proba, top_p=top_p, temperature=temperature, only_max=only_max, mode=mode,
                                guarantee=do_guarantee, time=c_time, T_time=t_time, det_BS=det_BS, ith=i)
                        
                else:   # Dont't guide                                    
                    for j in range(number_of_words_per_sentence):
                        context, proba, this_sequence, eos_c = sample_sentence_noguide(context, this_sequence, tokenizer, model, top_p=top_p, temperature=temperature, prev_proba=proba, eos_c=eos_c, det_BS=det_BS, ith=i)
                             
                if type(proba) == torch.Tensor:
                    proba = proba.item()
               
                score_ = get_score(k, number_of_words_per_sentence, online_probability[b], proba)
                w_c = number_keywords - len(guide_words)
                if not no_do_wc:
                    quality_score = evaluate_quality_linear(this_sequence, w_c, score_)    
                else:
                    quality_score = evaluate_quality_linear(this_sequence, 0, score_)    
                                
                # DEBUG:
                # print("Beam, Guidance: ", b, str(guide_words), guide[b])
                # print("txt, quality, wordC, score_: ", this_sequence, quality_score, w_c, score_)   

                # Linear Q             
                result_subsequences.append(
                        [context, quality_score, w_c, score_, online_probability[b]*proba, guide_words, guide[b], eos_c, guide_probs, t_time, c_time])
                
                if not guide[b]:
                    break   # No guiding, no multiple beams!
                              
            if k==0:        # First iteration of beam search is different!
                break
########################################################################################################
        # Deterministic K2T
        result_subsequences_sorted = sorted(result_subsequences, key=lambda a_entry: a_entry[1], reverse=True)      
              
        # Select Beams
        for b in range(number_of_beams):
            full_text[b] = result_subsequences_sorted[b][0]
            cum_quality_score[b] = result_subsequences_sorted[b][1]
            guide_words_s[b] = result_subsequences_sorted[b][5]
            guide_probs_s[b] = result_subsequences_sorted[b][8]
            guide[b] = result_subsequences_sorted[b][6]
            word_c[b] = result_subsequences_sorted[b][2]
            eos_count[b] = result_subsequences_sorted[b][7]
            total_time[b] = result_subsequences_sorted[b][9]
            current_time[b] = result_subsequences_sorted[b][10]
            if guide[b] and word_c[b] > number_keywords-1: # Only do this once, and then guide[b] no longer True
                guide[b] = False
                success_length[b] = k+1
            
            n_words_counter = (k+1)*number_of_words_per_sentence
            online_probability[b] = result_subsequences_sorted[b][4]
            online_perplexity = np.power(online_probability[b], (-1/n_words_counter))

            ### DEBUG: Comment out to remove console output
            # print(">>>>>>>>>>>>> BEAM: ", b)
            # print("Guidance words: ", keywords)
            # print("Current sentence: ", full_text[b])
            # print("Guidance word, word count, probs: ", guide_words_s[b], result_subsequences_sorted[b][2], guide_probs_s[b])
            # print("Current perplexity, cumulative quality, eos: ", online_perplexity, cum_quality_score[b], eos_count[b])        
            ###

            if np.sum(eos_count) == number_of_beams:
                print("Finishing...")
                break

        
        ''' Uncomment to write all intermediate steps to .txt

        text_file.write("\nBest 10 next subsequences: \n")
        for result_subsequence in result_subsequences_sorted:
            text_file.write(result_subsequence[0] + "\n Perplexity:" +
                            str(result_subsequence[2]) + "\n Quality Score: " +
                            str(result_subsequence[1]) + "\n\n")

        text_file.write("\n\n\n\n")
        '''
    #######################################
    # final evaluation
    #######################################
    end_time = time.time()
    time_needed = end_time - start_time

    success_length = get_success_length(success_length, number_of_beams, guide, number_of_generated_sentences)    
    success_rate = word_c[0]/number_keywords 
    distilGPT2_perplexity = distilGPT2_perplexity_score(full_text[0])

    ### Distinct n-grams
    sep = '<|endoftext|>'
    stripped = full_text[0].strip(sep).split(sep, 2)[0]
    tokenized_text = tokenizer.encode(stripped)
    # 2_Distinct
    counter_2 = Counter()
    total_2 = 0
    distinct_2 = 0   
    distinct_2, total_2, counter_2 = distinct_n(tokenized_text, 2, distinct_2, total_2, counter_2)      # Need to set n

    # 3_Distinct
    counter_3 = Counter()
    total_3 = 0
    distinct_3 = 0   
    distinct_3, total_3, counter_3 = distinct_n(tokenized_text, 3, distinct_3, total_3, counter_3)      # Need to set n

    # 4_Distinct
    counter_4 = Counter()
    total_4 = 0
    distinct_4 = 0   
    distinct_4, total_4, counter_4 = distinct_n(tokenized_text, 4, distinct_4, total_4, counter_4)      # Need to set n
    
    print("------------------------------------------------------------------------------")
    print("FINAL TEXT: ")
    print(full_text[0])
    print("Success rate, success length, perplexity: ", success_rate, success_length, distilGPT2_perplexity)

   
    #######################################
    # Save results, write in file and return
    #######################################

    # Declare evaluation
    evaluation = {
        "final_sequence: ": full_text[0],
        "keywords": keywords,
        #"online_perplexity": online_perplexity[0],
        "distilGPT2_perplexity": distilGPT2_perplexity,
        "success_rate": success_rate,
        "2_distinct": distinct_2,
        "2_total": total_2,
        "3_distinct": distinct_3,
        "3_total": total_3,
        "4_distinct": distinct_4,
        "4_total": total_4,
        "number_of_concurent_sentences": number_of_concurrent_sentences,
        "number_of_generated_sentences": number_of_generated_sentences,
        "number_of_words_per_sentence": number_of_words_per_sentence,
        "total_words": total_words,
        "top_k": top_k,
        "top_p": top_p,
        "model_name": model_name,
        "constant": constant,
        "time_needed": time_needed,
        "success_length": success_length,
        "guide_probs": guide_probs_s[0]
    }

    ### Write to text file
    text_file.write("Keywords: \n")
    for word in keywords:
        text_file.write(word + " ")
    text_file.write("\n\n")
    text_file.write("Final sequence: \n\n")
    text_file.write(full_text[0])
    for b in range(number_of_beams): 
        text_file_sentences.write(full_text[b])
        text_file_sentences.write("\n\n")
        text_file_sentences.write("\n\nSuccess_rate: " + str(word_c[b]/number_keywords))
        text_file_sentences.write("\nPerplexity: " + str(distilGPT2_perplexity_score(full_text[b])))
    text_file_sentences.write("\n###############################\n")
    text_file.write("\n\nSuccess_rate: " + str(success_rate))
    text_file.write("\nPerplexity: " + str(distilGPT2_perplexity))
    text_file.write("\nTime_needed: " + str(time_needed))
    text_file.write("\nSuccess_length: " + str(success_length))
    text_file.write("\n2_distint_rate: " + '{0:.4f}'.format(distinct_2/total_2))
    text_file.write("\n3_distint_rate: " + '{0:.4f}'.format(distinct_3/total_3))
    text_file.write("\n4_distint_rate: " + '{0:.4f}'.format(distinct_4/total_4))
    text_file.write("\n\n")
    text_file.close()
    text_file_sentences.close()
    
    del model
    torch.cuda.empty_cache()

    print("END: ", keywords)

    return evaluation

def get_folderfile_name(task, file_name):

    if task == 'key2article':
        folder_name = file_name + '/' 
    else:
        folder_name = os.path.dirname(file_name)
    
    
    abs_path = str(os.path.dirname(os.path.abspath(__file__)))
    file_name = str(os.path.abspath(os.path.join(abs_path, file_name)))
    folder_name = str(os.path.abspath(os.path.join(abs_path, folder_name)))
    if task == 'key2article':
        folder_name = folder_name + '/'
        file_name = file_name + '/' #Multiple files!
    print('file_name: ', file_name)
    print('folder_name: ', folder_name)
    
    return folder_name, file_name

def get_savefile(args):

    save_file = 'Result_w_'+str(args.weight)+'_nBeams_'+str(args.n_beams)+'_nGenSent_'+str(args.n_generated_sentences)+'_nWordsPerSent_'+str(args.n_words_per_sentence)+'_topP_'+str(args.top_p)
   
    if args.det_BS:
        save_file = save_file + '_detBS'
    if not args.no_do_wc:
        save_file = save_file + '_WC'
    if args.do_guarantee: 
        save_file = save_file + '_Guar_' + str(args.do_guarantee)
    if not args.guide:
        save_file = save_file + '_no_guide'
    if args.only_max == True:
        save_file = 'ONLYMAX_' + save_file        
    save_file = save_file + '_' + str(args.embedding)
    save_file = save_file + '_' + str(args.mode)
    
    return save_file

def get_savepath(task, results_subfolder, save_file, folder_name):

    if task == 'key2article':
        sub_folder = 'keyword_to_articles/' + str(results_subfolder) + '/'
        save_folder = 'results/' + sub_folder
        save_path = save_folder + save_file
    elif task == 'ROC':
        sub_folder = 'ROC/'
        save_folder = 'results/' + sub_folder
        save_path = save_folder + save_file
    elif task == 'commongen':
        sub_folder = 'commongen/'
        save_folder = 'results/' + sub_folder
        save_path = 'results/' + sub_folder + save_file
    else:
        sub_folder = os.path.basename(os.path.normpath(folder_name)) + '/' + str(results_subfolder) + '/'
        save_folder = 'results/' + sub_folder
        save_path = 'results/' + sub_folder + save_file

    try:
        os.mkdir(save_folder)
        print('made directory: ', save_folder)
    except OSError as error:
        print(error)
        
    return save_path
    
def get_keywordsets(task, folder_name, file_name):
    
    if task == 'key2article':
        keyword_sets = []
        for filename in os.listdir(folder_name):
            if filename.endswith('txt'):
                file1 = open(os.path.join(folder_name, filename), "r+")
                lines = file1.readlines()
                keywords = list(lines[2].strip().split(", "))
                in_text = lines[1].split()[:30]
                keyword_sets.append((' '.join(in_text), keywords))
    else:
        #File containing the keywords as text
        in_text = '<|endoftext|>' # 'Start with EOS
        #in_texts = ['I', 'It', 'A'] #Other possible start tokens
        file1 = open(file_name, "r+")
        lines = file1.readlines()
        if task == 'commongen':
            keyword_sets = [(in_text, list(line.strip().split())) for line in lines]
        else:
            keyword_sets = [(in_text, list(line.strip().split(", "))) for line in lines]
            # keyword_sets = [(random.choice(in_texts), list(line.strip().split(", "))) for line in lines]

    return keyword_sets
    
    
def get_args(parser):

    # Get constant defined in run_gpt2.sh
    # Default is GPT-3 Beam Search except det_BS

    parser.add_argument('-top_p', type=float, default=0.9)
    parser.add_argument('-weight', type=float, default=5.0) #20.0
    parser.add_argument('-n_generated_sentences', type=int, default=90)
    parser.add_argument('-n_words_per_sentence', type=int, default=1)
    parser.add_argument('-n_beams', type=int, default=1)
    parser.add_argument('-n_repetitions', type=int, default=1)
    parser.add_argument('-temperature', type=float, default=1.)
    parser.add_argument('-only_max', type=bool, default=False)
    parser.add_argument('-no_do_wc', type=bool, default=False)  
    parser.add_argument('-mode', type=str, default='max',
                        choices=['max', 'next', 'all', 'random', 'best_tour'], help='modes: max, next, all, random, best_tour')
    parser.add_argument('-do_guarantee', type=bool, default=False)
    parser.add_argument('-embedding', type=str, default='glove',
                        choices=list(word_embedding.keys()), help='word_embedding') 
    parser.add_argument('-file_name', type=str, default='data/50_keywordsets_eval/word_sets.txt')  #data/50_keywordsets_eval/word_sets data/commongen_small/commongen.dev.src_alpha_small.txt
    parser.add_argument('-det_BS', type=bool, default=False)
    parser.add_argument('-guide', type=bool, default=True)
    parser.add_argument('-results_subfolder', type=str, default='tmp')
    parser.add_argument('-task', type=str, default='50keywords',
                        choices=['50keywords', 'ROC', 'key2article', 'commongen'], help='tasks: 50keywords, ROC, key2article, commongen')
    args = parser.parse_args()

    return args
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = get_args(parser)

    file_name = args.file_name
    file_name = file_name.strip('/')    
    if not file_name:
        raise Exception("file_name name missing. Please give the relative path to word_sets filename (or the word_sets folder in case of key2article flag is True).")
    
    ### Create model
    model = GPT2LMHeadModel.from_pretrained('gpt2-large')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    model.eval()   
    model.to('cuda') #

    # Get keywords and save path  
    folder_name, file_name = get_folderfile_name(args.task, file_name)
    save_file = get_savefile(args)
    save_path = get_savepath(args.task, args.results_subfolder, save_file, folder_name)     
    keyword_sets = get_keywordsets(args.task, folder_name, file_name)
    
    print('Mode:', args.mode)
    print('Save path: ', save_path)
           
    # Create file containing the keyword embeddings
    save_path_dict = os.path.join(folder_name, 'dict_' + str(args.embedding) + '.pkl')
    if not os.path.isfile(save_path_dict):
        create_enc_dict(file_name, args.embedding, task=args.task)
    with open(save_path_dict, 'rb') as f:
        enc_dict = pickle.load(f)

    ################ RUN ################
    all_results = np.zeros([len(keyword_sets), args.n_repetitions, 11], dtype = object) # Initialize results structure
    
    for j, keyword_set in enumerate(keyword_sets):
    
        if args.n_generated_sentences<0:
            in_text, keywords = keyword_set
            n_of_generated_sentences = math.ceil((len(keywords)+1) * abs(args.n_generated_sentences) / args.n_words_per_sentence)
        else:
            n_of_generated_sentences = args.n_generated_sentences
            
        for i in range(args.n_repetitions):
            results = conditional_language_generation(model,tokenizer,  
                                                        keyword_set=keyword_set,
                                                        top_p=args.top_p,
                                                        constant=args.weight,
                                                        number_of_concurrent_sentences=args.n_beams,
                                                        number_of_generated_sentences=n_generated_sentences,
                                                        number_of_words_per_sentence=args.n_words_per_sentence,
                                                        number_of_beams = args.n_beams,
                                                        enc_dict=enc_dict, 
                                                        save_path=save_path, 
                                                        temperature=args.temperature,
                                                        only_max=args.only_max,
                                                        no_do_wc=args.no_do_wc,
                                                        mode=args.mode,
                                                        do_guarantee=args.do_guarantee,
                                                        embedding=args.embedding,
                                                        folder_name=folder_name,
                                                        det_BS=args.det_BS,
                                                        guide=args.guide,
                                                        )
            all_results[j][i][0] = results["distilGPT2_perplexity"]
            all_results[j][i][1] = results["time_needed"]
            all_results[j][i][2] = results["success_rate"]
            all_results[j][i][3] = results["success_length"]   
            all_results[j][i][4] = results["2_distinct"]   
            all_results[j][i][5] = results["2_total"]   
            all_results[j][i][6] = results["3_distinct"]
            all_results[j][i][7] = results["3_total"] 
            all_results[j][i][8] = results["4_distinct"]
            all_results[j][i][9] = results["4_total"] 
            all_results[j][i][10] = results["guide_probs"]     
        
    np.save(save_path, all_results)

    
    
