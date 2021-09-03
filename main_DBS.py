import time
import torch
import json
import os
import numpy as np
import scipy.io as sio
import argparse
import gc

os.environ['TRANSFORMERS_CACHE']='./transformer_models'  # Work-around to avoid memory problems in server, comment out depending on memory availability

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


#import gensim.downloader as api
#word_vectors = api.load("glove-wiki-gigaword-300")

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

def sample_sentence(text, this_sequence, tokenizer, model, keywords, enc_dict, guide_probs, converter_table, weight, guide=False, prev_proba=1, top_k=0, top_p=0.9, temperature=1., only_max=False, mode='max', guarantee=False, time=0, T_time=1, det_BS=False, ith=0, force_word=False):
    """ Samples the next word of the sequence with logit modification (guidance)
    Modes:
        mode='max':     each token is shifted by the cosine similarity to the closest guide word
        mode='all':     each token is shifted by the cosine similarity to each guide word
        mode='next':    the order of the guide words is fixed and each token is shifted towards the next guide word in the sequence
        mode='random':  a random word is selected from the remaining (not yet appeared) guide words and each token is shifted towards this guide word
    """    

    # print('T_time: ', time, T_time)
    # print("DEB_SS: ", text, keywords, guide)
    guide_word_stems = [porter.stem(w.lower()) for w in keywords]
    keywords_ = [w for w in keywords]

    # Select the next guide word(s)
    if keywords_:
        if mode=='next':
            keywords_ = [keywords_[0]]
        if mode=='random':
            keywords_ = [random.choice(keywords_)]

    # print('keywords_: ', keywords, keywords_)

    keywords_enc = [enc_dict[w] for w in keywords_]
    keywords_gpt = {tokenizer.encode(w)[0]:w for w in keywords_}

    # print('keywords_gpt: ', keywords_gpt)

    # print('mode:', mode, keywords, len(keywords_enc))

    ## GPT2 - generate logits
    indexed_tokens = tokenizer.encode(text)
    indexed_this_seq = tokenizer.encode(this_sequence)
    tokens_tensor = torch.tensor([indexed_tokens])
    tokens_tensor = tokens_tensor.to('cuda')
    #model.to('cuda')
    
    # Predict all tokens
    #with torch.no_grad():
    outputs = model(tokens_tensor)
    del tokens_tensor
    torch.cuda.empty_cache()

    logits = outputs.logits
    logits = logits[0, -1, :]/ temperature

    # logits = F.softmax(logits, dim=-1)  
    # weight  = weight *0.03  
    
    proba = F.softmax(logits, dim=-1) 
    logits = F.log_softmax(logits, dim=-1)  
    # print('proba vs logit: ', (max(proba)/max(logits)).item(), max(logits).item(), sum(logits).item()) 
    
    # Calculate cosine similarity
    if keywords_enc and guide:
        if len(keywords_enc)>1:
            sims = np.array([cosine_similarity(np.reshape(w, (1, -1)), converter_table) for w in keywords_enc])
            # print(sims.shape)
            if force_word:
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

        if only_max == True:
            sim_aux = np.zeros_like(sim)
            sim_aux[0,sim.argmax()] = sim.max()
            sim = np.squeeze(sim_aux)
        else:
            sim = np.clip(np.squeeze(sim), a_min=0, a_max=None) #tf.square(sim)  
        # sim = sim*sim     ###

        if guarantee:
            if T_time == 0:
                T_time = 1
            rate = (1/T_time)*np.log(100/weight)  # 100 is the maximum value the weight will reach it may affect ppl, with 50 ppl is around 8 but fails to ALWAYS reach 100% Success rate
            weight = weight*np.exp(rate*time)
            # print('weight: ', weight)
        
        logits = logits + torch.tensor(sim*weight).cuda()

    ##logits = F.softmax(logits, dim=-1) 

    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p) ###  
    logits = F.softmax(logits, dim=-1)  ###
        
    if det_BS:
        predicted_index = torch.topk(logits, ith+1)[1][ith].item()
        # print("Pred index: ", predicted_index, ith)
    else:
        predicted_index = torch.multinomial(logits, 1).item()

    if guarantee and time > T_time:
        # predicted_index = random.choice(list(keywords_gpt.keys()))
        predicted_index = list(keywords_gpt.keys())[0]
        print('FORCE: ', predicted_index, keywords_gpt[predicted_index])
    
    if force_word and predicted_index in keywords_gpt:
        predicted_text = tokenizer.decode(indexed_tokens) + ' ' + keywords_gpt[predicted_index]
        this_sequence = tokenizer.decode(indexed_this_seq) + ' ' + keywords_gpt[predicted_index]
        pred_word = keywords_gpt[predicted_index]
        print('force rest of word: ', pred_word, tokenizer.decode(predicted_index))
    else:
        predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
        this_sequence = tokenizer.decode(indexed_this_seq + [predicted_index])
        #pred_word = predicted_text.split()[-1]
        pred_word = predicted_text.split()[-1].split('<|endoftext|>')[-1]
    
    pred_word_stem = porter.stem(pred_word.lower())
    
    guide_next = guide
    time_next = time+1
    T_time_next = T_time
    if pred_word_stem in guide_word_stems:
        ind = guide_word_stems.index(pred_word_stem)
        # print(pred_word_stem, guide_word_stems, keywords)
        keywords = keywords[:ind] + keywords[ind+1:]
        guide_probs = guide_probs + [(pred_word_stem, proba[predicted_index].item())]
        # print(pred_word_stem, guide_word_stems, keywords)
        guide_next = False
        time_next = 1
        T_time_next = T_time-time+1
        

    return predicted_text, keywords, guide_next, guide_probs, prev_proba*proba[predicted_index], this_sequence, time_next, T_time_next

    # if pred_word_stem == guide_word_stem:
    #     guide_next = False
    #     word_count_next = 1
    # else:
    #     guide_next = guide
    #     word_count_next = word_count

    #if predicted_index == 50256:
    #    print("END OF TEXT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # return predicted_text, guide_next, prev_proba*proba[predicted_index].item(), this_sequence, word_count_next

def sample_sentence_eos(text, this_sequence, tokenizer, model, prev_proba=1, top_k=0, top_p=0.9, temperature=1., eos_c=0, det_BS=False, ith=0):
    """ Samples the next word of the sequence without logit modification (guidance)
    """
    ## GPT2 - generate logits
    indexed_tokens = tokenizer.encode(text)
    indexed_this_seq = tokenizer.encode(this_sequence)
    tokens_tensor = torch.tensor([indexed_tokens])
    tokens_tensor = tokens_tensor.to('cuda')
    #model.to('cuda')

    # Predict all tokens
    outputs = model(tokens_tensor)
    del tokens_tensor
    torch.cuda.empty_cache()

    logits = outputs.logits
    logits = logits[0, -1, :]/ temperature
    proba = F.softmax(logits, dim=-1)        
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    #print("Max logits after sampling: ", logits.shape, torch.max(logits))
    
    #if eos_c == 0:
    #    logits[50256] = logits[50256] + torch.tensor(0.5*weight).cuda()
    

    logits = F.softmax(logits, dim=-1) 

    if det_BS:
        predicted_index = torch.topk(logits, ith+1)[1][ith].item()
        # print("Pred index: ", predicted_index, ith)
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


def sample_sentence_noguide(text, this_sequence, tokenizer, model, prev_proba=1, top_k=0, top_p=0.9, temperature=1., det_BS=False, ith=0):
    """ Samples the next word of the sequence without logit modification (guidance
    """
    ## GPT2 - generate logits
    indexed_tokens = tokenizer.encode(text)
    indexed_this_seq = tokenizer.encode(this_sequence)
    tokens_tensor = torch.tensor([indexed_tokens])
    tokens_tensor = tokens_tensor.to('cuda')
    #model.to('cuda')

    # Predict all tokens
    outputs = model(tokens_tensor)
    del tokens_tensor
    torch.cuda.empty_cache()

    logits = outputs.logits
    logits = logits[0, -1, :]/ temperature
    proba = F.softmax(logits, dim=-1)        
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    #print("Max logits after sampling: ", logits.shape, torch.max(logits))
    
    logits = F.softmax(logits, dim=-1) 

    if det_BS:
        predicted_index = torch.topk(logits, ith+1)[1][ith].item()
    else:
        predicted_index = torch.multinomial(logits, 1).item()
    
    predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
    this_sequence = tokenizer.decode(indexed_this_seq + [predicted_index])
    pred_word = predicted_text.split()[-1]  
    
    return predicted_text, prev_proba*proba[predicted_index].item(), this_sequence

def conditional_language_generation(
    keyword_set,
    model_name='gpt2-large',
    enc_dict={},
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=0.9,
    models_dir='models',
    constant=20,
    number_of_concurrent_sentences = 10,
    number_of_generated_sentences = 20,
    number_of_words_per_sentence = 5,
    number_of_beams = 3,
    word_index=0,
    save_path='dummy.txt',
    sample=False,
    temp=1.,
    only_max = False,
    eos=False,
    do_Q_logprob=False,
    do_Q_logprob_norm=False,
    no_do_wc=False,
    do_wc_eos=False,
    mode='max',
    do_guarantee=False,
    embedding='glove',
    det_BS=False,
    force_word=False,
    folder_name='',
    guide=True
):
    """
    Main function for conditional language generation
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
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

    #Define model: here GPT2 large from the Hugging Face
    model = GPT2LMHeadModel.from_pretrained('gpt2-large')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    model.eval()
    model.to('cuda')
    
###################################
	## Load words

    #length = number_of_words_per_sentence
    
    # Define task, keyword to article, keyword to story (ROC) or keyword to phrase
    in_text, keywords = keyword_set
    
    keywords_enc = [enc_dict[w] for w in keywords]
    print(in_text, keywords)
    full_text = [in_text] * number_of_beams 
    # print("Full text: ", full_text)

    if mode=='best_tour':
        best_order = best_tour(keywords_enc)
        keywords_enc = [keywords_enc[i] for i in list(best_order)]
        print("Keywords: ", keywords, best_order)
        keywords = [keywords[i] for i in best_order]
        print("Keywords: ", keywords)
        mode = 'next' # Switch over to next (ordered) mode with the optimized order
    if mode=='worst_tour':
        best_order = best_tour(keywords_enc, distance=pos_cosine_similarity)
        keywords_enc = [keywords_enc[i] for i in list(best_order)]
        print("Keywords: ", keywords, best_order)
        keywords = [keywords[i] for i in best_order]
        print("Keywords: ", keywords)
        mode = 'next' # Switch over to next (ordered) mode with the optimized order

    number_keywords = len(keywords)
    print("N keywords: ", number_keywords)
###################################
    from datetime import datetime
    # To measure computation time
    now = datetime.now()
    # File to save results as .txt
    text_file = open(save_path + '.txt', 'a+', encoding='utf8')
    text_file_sentences = open(save_path + 'SENTENCES.txt', 'a+', encoding='utf8')
       
    # prepare variables...
    #np.random.seed(seed)
    weight = constant
    converter_table = np.load(str(os.path.dirname(
        os.path.abspath(__file__))) + '/data/converter_table_' + str(embedding) + '.npy')  
    
    
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
    
    alpha = 0.6
    for k in range(number_of_generated_sentences):      

        # Define guidance word and index for guidance in model and quality function     
        result_subsequences = []


        for b in range(number_of_beams):

            # Reset variables:
            # beam_text = full_text[b]
            # guide_words = guide_words_s[b]
            # print("Guidance: ", str(guide_words))
            # guide_word_stem = porter.stem(guidance_word.lower())
            
            #perplexities = np.zeros((number_of_concurrent_sentences))

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
                                weight, guide_next, proba, top_p=top_p, only_max=only_max, mode=mode,
                                guarantee=do_guarantee, time=c_time, T_time=t_time, det_BS=det_BS, ith=i, force_word=force_word)
                        # context, guide_next, proba, this_sequence, w_c = sample_sentence(context, this_sequence, tokenizer, model, 
                        #         guide_word_stem, keywords_enc[guidance_index[b]], converter_table, weight, w_c, guide_next, proba, top_p=top_p, only_max=only_max)
                else:   # Dont't guide                    
                    if eos == False:
                        for j in range(number_of_words_per_sentence):
                            context, proba, this_sequence = sample_sentence_noguide(context, this_sequence, tokenizer, model, top_p=top_p, prev_proba=proba, det_BS=det_BS, ith=i)
                    else:
                        for j in range(number_of_words_per_sentence):
                            context, proba, this_sequence, eos_c = sample_sentence_eos(context, this_sequence, tokenizer, model, top_p=top_p, prev_proba=proba, eos_c=eos_c, det_BS=det_BS, ith=i)
                

                w_c = number_keywords - len(guide_words)
                # print("w_c: ", w_c, number_keywords, len(guide_words), guide_next)

                if type(proba) == torch.Tensor:
                    proba = proba.item()

                # Exp Q:
                #perplexity = np.power(proba, (-1/length))                
                #quality_score, word_count = evaluate_quality(this_sequence, guidance_word, 0, perplexity, guide[b], temp)
                # Linear Q: (corrected version)
                
                length = (k+1)*number_of_words_per_sentence 
                if do_Q_logprob:
                    perplexity = np.log(online_probability[b]*proba)
                elif do_Q_logprob_norm:
                    len_norm = ((5+length)**alpha)/(6**alpha)
                    perplexity = np.log(online_probability[b]*proba)/len_norm
                else: #ppl
                    perplexity = -np.power(online_probability[b]*proba, (-1/(length*(k+1))))     # Total probability, total length

                if not no_do_wc:
                    w_c_score = w_c # word_c[b]+w_c
                else:
                    w_c_score = 0
                if do_wc_eos:
                    w_c_score = w_c_score + eos_c


                if do_Q_logprob:
                    quality_score = evaluate_quality_linear(this_sequence, w_c_score, perplexity, temp)   
                elif do_Q_logprob_norm:
                    quality_score = evaluate_quality_linear(this_sequence, w_c_score, perplexity, temp)    
                else:
                    quality_score = evaluate_quality_linear(this_sequence, w_c_score, perplexity, temp, perp=True)            

                # DEBUG:
                # print("Beam, Guidance: ", b, str(guide_words), guide[b])
                # print("txt, quality, wordC, ppl: ", this_sequence, quality_score, w_c, perplexity)   

                # Linear Q             
                result_subsequences.append(
                        [context, quality_score, w_c, perplexity, online_probability[b]*proba, guide_words, guide[b], eos_c, guide_probs, t_time, c_time])
                
                if not guide[b]:
                    break #No guiding, no multiple beams!
                    
                #perplexities[i] = perplexity            
            if k==0:        # First iteration of beam search is different!
                break
########################################################################################################
        # Deterministic K2T
        if not sample:
            result_subsequences_sorted = sorted(
                result_subsequences, key=lambda a_entry: a_entry[1], reverse=True)      
        # Sample K2T
        else:
            scores = torch.tensor([a_entry[1] for a_entry in result_subsequences])
            #print("Scores: ", scores)
            soft_scores = F.softmax(scores, dim=-1) 
            #print("Soft scores: ", soft_scores)
            sampled_indeces = torch.multinomial(soft_scores, len(result_subsequences), replacement=False).tolist()
            #print("Sampled indeces: ", sampled_indeces)
            result_subsequences_sorted = [result_subsequences[i] for i in sampled_indeces]
            print(result_subsequences_sorted[0])
            del sampled_indeces
            del soft_scores
            torch.cuda.empty_cache()       

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
            # Exp Q:
            """
            if result_subsequences_sorted[b][2] > 0: ## Word Count
                guidance_index[b] += 1
                if guidance_index[b] > number_keywords-1:
                    guide[b] = False
                    guidance_index[b] = 0
                    success_length[b] = k+1
            """
            n_words_counter = (k+1)*number_of_words_per_sentence
            online_probability[b] = result_subsequences_sorted[b][4]
            online_perplexity = np.power(online_probability[b], (-1/n_words_counter))

            # DEBUG: Comment to remove console output
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
    for b in range(number_of_beams):    
        if guide[b]:
            success_length[b] = 0

    # Success rate
    target_words = number_keywords
    target_count = 0
    for i in range(number_keywords):
        if count_word_stem(keywords[i], full_text[0]) > 0:
            target_count += 1
            
    success_rate = word_c[0]/number_keywords #target_count/target_words

    # Distil-GPT2 perplexity
    distilGPT2_perplexity = distilGPT2_perplexity_score(full_text[0])

    ### Distinct n-grams
    sep = '<|endoftext|>'
    # print('full_text[0]: ', full_text[0])
    # print(full_text[0].split(sep, 2))
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
    print("Success rate, success length, perplexity: ", success_rate, success_length[0]/number_of_generated_sentences, distilGPT2_perplexity)
    # Time measurement
    
    

    # Save evaluations

    # declare evaluations
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
        "success_length": success_length[0]/number_of_generated_sentences,
        "guide_probs": guide_probs_s[0]
    }

    # Write to text file
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
    text_file.write("\nSuccess_length: " + str(success_length[0]))
    text_file.write("\n2_distint_rate: " + '{0:.4f}'.format(distinct_2/total_2))
    text_file.write("\n3_distint_rate: " + '{0:.4f}'.format(distinct_3/total_3))
    text_file.write("\n4_distint_rate: " + '{0:.4f}'.format(distinct_4/total_4))
    # text_file.write("\nGuide_probs: " + str(guide_probs_s[0]))
    text_file.write("\n\n")

    text_file.close()
    text_file_sentences.close()
    

    del model
    torch.cuda.empty_cache()

    print("END: ", keywords)

    return evaluation


if __name__ == '__main__':

    # Get constant defined in run_gpt2.sh
    # Default is GPT-3 Beam Search except det_BS
    parser = argparse.ArgumentParser()
    parser.add_argument('-top_p', type=float, default=0.9)
    parser.add_argument('-weight', type=float, default=5.0) #20.0
    parser.add_argument('-n_generated_sentences', type=int, default=90)
    parser.add_argument('-n_words_per_sentence', type=int, default=1)
    parser.add_argument('-n_beams', type=int, default=1)
    parser.add_argument('-n_repetitions', type=int, default=1)
    parser.add_argument('-sample', type=bool, default=False)
    parser.add_argument('-temperature', type=float, default=1.)
    parser.add_argument('-only_max', type=bool, default=False)
    parser.add_argument('-key2article', type=bool, default=False)
    parser.add_argument('-ROC', type=bool, default=False)
    parser.add_argument('-eos', type=bool, default=True)
    parser.add_argument('-no_do_wc', type=bool, default=False)
    parser.add_argument('-do_Q_logprob', type=bool, default=False)    
    parser.add_argument('-do_Q_logprob_norm', type=bool, default=True)    
    parser.add_argument('-do_wc_eos', type=bool, default=False)
    parser.add_argument('-mode', type=str, default='max',
                        choices=['max', 'next', 'all', 'random', 'best_tour', 'worst_tour'], help='modes: max, next, all, random, best_tour, worst_tour')
    parser.add_argument('-do_guarantee', type=bool, default=False)
    parser.add_argument('-word_embedding', type=str, default='glove',
                        choices=list(word_embedding.keys()), help='word_embedding') 
    parser.add_argument('-file_name', type=str, default='data/50_keywordsets_eval/word_sets.txt')  #data/50_keywordsets_eval/word_sets data/commongen_small/commongen.dev.src_alpha_small.txt
    parser.add_argument('-det_BS', type=bool, default=False)
    parser.add_argument('-force_word', type=bool, default=True)
    parser.add_argument('-guide', type=bool, default=True)
    parser.add_argument('-results_subfolder', type=str, default='tmp')
    parser.add_argument('-task', type=str, default='50keywords',
                        choices=['50keywords', 'commongen'], help='tasks: 50keywords, commongen')
    args = parser.parse_args()

    #random.seed(42)
    #torch.manual_seed(42)
    #np.random.seed(42)
    
    top_p=args.top_p
    weight = args.weight
    number_of_concurrent_sentences = args.n_beams
    n_generated_sentences = args.n_generated_sentences
    number_of_words_per_sentence = args.n_words_per_sentence
    number_of_beams = args.n_beams
    sample = args.sample
    n_repetitions = args.n_repetitions
    temperature = args.temperature
    only_max = args.only_max
    key2article = args.key2article
    ROC = args.ROC
    eos = args.eos
    do_Q_logprob = args.do_Q_logprob
    do_Q_logprob_norm = args.do_Q_logprob_norm
    no_do_wc = args.no_do_wc
    do_guarantee = args.do_guarantee
    embedding = args.word_embedding
    det_BS = args.det_BS
    force_word = args.force_word
    guide = args.guide
    results_subfolder = args.results_subfolder

    do_wc_eos = args.do_wc_eos
    mode = args.mode
    file_name = args.file_name
    file_name = file_name.strip('/')

    task = args.task

    if not file_name:
        raise Exception("file_name name missing. Please give the relative path to word_sets filename (or the word_sets folder in case of key2article flag is True).")

    if key2article:
        folder_name = file_name + '/'
    else:
        folder_name = os.path.dirname(file_name)

    abs_path = str(os.path.dirname(os.path.abspath(__file__)))
    file_name = str(os.path.abspath(os.path.join(abs_path, file_name)))
    print('folder_name2: ', folder_name)
    folder_name = str(os.path.abspath(os.path.join(abs_path, folder_name)))
    if key2article:
        folder_name = folder_name + '/'
        file_name = file_name + '/'
    print('file_name: ', file_name)
    print('folder_name: ', folder_name)

    # Deterministic or sample K2T
    if sample == False:
        save_file = 'Break_LinearQ_noSq_deterministic_result_w_'+str(weight)+'_nBeams_'+str(number_of_beams)+'_nConcSent_'+str(number_of_concurrent_sentences)+'_nGenSent_'+str(n_generated_sentences)+'_nWordsPerSent_'+str(number_of_words_per_sentence)+'_topP_'+str(top_p)
    else:
        save_file = 'LinearQ_noSq_sample_result_w_'+str(weight)+'_nBeams_'+str(number_of_beams)+'_nConcSent_'+str(number_of_concurrent_sentences)+'_nGenSent_'+str(n_generated_sentences)+'_nWordsPerSent_'+str(number_of_words_per_sentence)+'_temperature_'+str(temperature)+'_topP_'+str(top_p)

    if do_Q_logprob:
        save_file = save_file + '_QLogP'
    elif do_Q_logprob_norm:
        save_file = save_file + '_QLogPNorm'
    else:
        save_file = save_file + '_QPPL'  
    if det_BS:
        save_file = save_file + '_detBS'
    if not no_do_wc:
        save_file = save_file + '_WC'
    if do_wc_eos:
        save_file = save_file + '_WCEOS'
    if eos: 
        save_file = save_file + '_EOS'
    if do_guarantee: 
        save_file = save_file + '_Guar_' + str(do_guarantee)
    if force_word: 
        save_file = save_file + '_force_' + str(force_word)
    if not guide:
        save_file = save_file + '_no_guide'
    save_file = save_file + '_' + str(embedding)
    # save_file = save_file + '_' + str('beni')
    save_file = save_file + '_' + str(mode)

    # K2T-one or not
    if only_max == True:
        save_file = 'ONLYMAX_' + save_file

    #Task
    if key2article:
        sub_folder = 'keyword_to_articles/' + str(results_subfolder) + '/'
        save_folder = 'results/' + sub_folder
        save_path = save_folder + save_file
    elif ROC:
        sub_folder = 'ROC/'
        save_folder = 'results/' + sub_folder
        save_path = save_folder + save_file
    else:
        sub_folder = os.path.basename(os.path.normpath(folder_name)) + '/' + str(results_subfolder) + '/'
        save_folder = 'results/' + sub_folder
        save_path = save_folder + save_file
        save_path = 'results/' + sub_folder + save_file

    if task == 'commongen':
        sub_folder = 'commongen_small/'
        save_path = 'results/' + sub_folder + save_file
    # else:
    #     sub_folder = os.path.basename(os.path.normpath(folder_name)) + '/final/'
    #     save_path = 'results/' + sub_folder + save_file
    #save_path = 'results/'
    try:
        os.mkdir(save_folder)
        print('made directory: ', save_folder)
    except OSError as error:
        print(error)
    
    print('mode:', mode)
    print('Save path: ', save_path)


    if key2article:
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
        in_text = '<|endoftext|>' # 'It is' is arbitrary, we should start with EOS
        in_texts = ['I', 'It', 'A']
        file1 = open(file_name, "r+")
        lines = file1.readlines()
        if task == 'commongen':
            print("COMMONGEN")
            print(lines[0].strip())
            print(lines[0].strip().split())
            keyword_sets = [(in_text, list(line.strip().split())) for line in lines]
            c_gen = True
        else:
            keyword_sets = [(in_text, list(line.strip().split(", "))) for line in lines]
            # keyword_sets = [(random.choice(in_texts), list(line.strip().split(", "))) for line in lines]
            c_gen = False

    #File containing the keyword embeddings
    save_path_dict = os.path.join(folder_name, 'dict_' + str(embedding) + '.pkl')
    if not os.path.isfile(save_path_dict):
        create_enc_dict(file_name, embedding, key2article=key2article, commongen=c_gen)
    with open(save_path_dict, 'rb') as f:
        enc_dict = pickle.load(f)

    all_results = np.zeros([len(keyword_sets), n_repetitions, 11], dtype = object)
    # For every concept word set

    ## 50 keywords 
    for j, keyword_set in enumerate(keyword_sets):
        if n_generated_sentences<0:
            in_text, keywords = keyword_set
            number_of_generated_sentences = math.ceil((len(keywords)+1) * abs(n_generated_sentences) / number_of_words_per_sentence)
        else:
            number_of_generated_sentences = n_generated_sentences
        print("number_of_generated_sentences: ", number_of_generated_sentences, number_of_words_per_sentence)
        for i in range(n_repetitions):
            results = conditional_language_generation(keyword_set=keyword_set,
                                                        top_p=top_p,
                                                        constant=weight,
                                                        number_of_concurrent_sentences=number_of_concurrent_sentences,
                                                        number_of_generated_sentences=number_of_generated_sentences,
                                                        number_of_words_per_sentence=number_of_words_per_sentence,
                                                        number_of_beams = number_of_beams,
                                                        enc_dict=enc_dict, 
                                                        save_path=save_path, 
                                                        sample=sample,
                                                        temp=temperature,
                                                        only_max=only_max,
                                                        eos=eos,
                                                        do_Q_logprob=do_Q_logprob,
                                                        do_Q_logprob_norm=do_Q_logprob_norm,
                                                        no_do_wc=no_do_wc,
                                                        do_wc_eos=do_wc_eos,
                                                        mode=mode,
                                                        do_guarantee=do_guarantee,
                                                        embedding=embedding,
                                                        folder_name=folder_name,
                                                        det_BS=det_BS,
                                                        force_word=force_word,
                                                        guide=guide,
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

    
    
