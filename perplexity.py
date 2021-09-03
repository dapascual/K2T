import math
import torch
import os
os.environ['TRANSFORMERS_CACHE']='.'

from transformers import GPT2TokenizerFast, GPT2LMHeadModel
# Load pre-trained model (weights)
model = GPT2LMHeadModel.from_pretrained('distilgpt2')
model.eval()
# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2TokenizerFast.from_pretrained('distilgpt2')


def distilGPT2_perplexity_score(sentence):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor(
        [tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss, logits = model(tensor_input, labels=tensor_input)[:2]

    return math.exp(loss)


