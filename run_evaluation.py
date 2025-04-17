# Import stuff
import plotly.io as pio
import plotly.graph_objects as go
import accelerate
pio.renderers.default = "colab"
import transformer_lens
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from fancy_einsum import einsum
import tqdm.notebook as tqdm
import random
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader

from jaxtyping import Float, Int
from torch import Tensor
from typing import List, Union, Optional, Tuple
from functools import partial
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML
import circuitsvis as cv

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens import HookedTransformer, HookedTransformerConfig, ActivationCache


model = HookedTransformer.from_pretrained(
    'pythia-70m',
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    low_cpu_mem_usage=True
) #transformer lens automatically assigns the model to a device, but set a variable for later
device = 'cuda:0' if torch.cuda.is_available() else 'cpu' #set the device if a GPU is available

example_prompt = " a d c z u"*3
example_answer = " a"
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)

#These are some examples to help you later
prompt = example_prompt

#Get all of the tokens for this prompt
#Note: You can also pass a list of prompts to tokenize all of them
out = model.to_tokens(prompt)
print("Prompt tokens\n", out) #50256 is the BOS token (The EOS token is used for this)


#Get a single token for some word, like " red"
s = " red"
out = model.to_single_token(s)
print("Single token\n", out)

#Get the tokens for the prompt decoded as strings
out = model.to_str_tokens(prompt, prepend_bos=False)
print("String tokens, no BOS:\n", out)

#get the tokens for BOTH prompts at the same time, with BOS
tokens = model.to_tokens(example_prompt, prepend_bos=True)
# Move the tokens to the GPU
tokens = tokens.to(device)
# Run the model and cache all activations
original_logits, cache = model.run_with_cache(tokens)


def get_attn_pattern(cache, layer) -> torch.Tensor:
  #layer: the int layer index of the model we are targeting
  return cache[utils.get_act_name(name='pattern', layer=layer)]


def get_mlp_post(cache, layer) -> torch.Tensor:
  #layer: the int layer index of the model we are targeting
  return cache[utils.get_act_name(name='mlp_post', layer=layer)]

def get_residual_before_attn(cache, layer) -> torch.Tensor:
  #layer: the int layer index of the model we are targeting
  return cache[utils.get_act_name(name='resid_pre', layer=layer)]


target_layer = 3

attn_pattern = get_attn_pattern(cache, target_layer)
print("Attn pattern shape", attn_pattern.shape)


mlp_post = get_mlp_post(cache, target_layer)
print("MLP Post shape", mlp_post.shape)

residual_pre_attn = get_residual_before_attn(cache, target_layer)
print("Residual shape", residual_pre_attn.shape)

ex_tokens = model.to_tokens(example_prompt, prepend_bos=False)
ex_logits, ex_cache = model.run_with_cache(ex_tokens, remove_batch_dim=True, prepend_bos=False)
ex_str_tokens = model.to_str_tokens(example_prompt, prepend_bos=False)
print(example_prompt, ex_str_tokens)

layer_idx=3
attention_pattern = ex_cache["pattern", layer_idx, "attn"]
print(attention_pattern.shape)

print(f"Layer {layer_idx} Head Attention Patterns:")
cv.attention.attention_patterns(tokens=ex_str_tokens, attention=attention_pattern)

#induction head is an attention head where token i+1 attends to token i in the previous sequence
ind_layer_idx = 3
ind_head_idx = 6


idx = 5
# all_tokens = model.to_tokens(dataset, prepend_bos=False)

def calc_accuracy(model, dataset, labels) -> float:
  #Calculate the accuracy of the model on the random tokens dataset you made
  #return it as a float. You should get ~100% accuracy

  #TODO
  logits, cache = model.run_with_cache(dataset, prepend_bos=True)
  probs = torch.nn.functional.softmax(logits, dim=-1)[:,-1,:]
  pred_label = torch.argmax(probs, axis=-1)

  true_label = model.to_tokens(labels, prepend_bos=False)
  true_label = torch.squeeze(true_label)


  count = (pred_label == true_label).sum().item()

  return count / len(dataset)

acc = calc_accuracy(model, dataset, labels)
print("Accuracy:", acc)


pattern = cache['pattern', ind_layer_idx, 'attn'] #you can omit 'attn' here, it also works
pattern.shape #batch, num_heads, dest, source (attend from, attend to)
head_pattern = pattern[:, ind_head_idx]
print(head_pattern.shape)

#Visualize a single attention pattern (sort of like above) for a given head and example

example = head_pattern[0]
print(example.shape)
px.imshow(example.cpu().numpy())

def reshape_heatmap(patterns) -> Union[torch.Tensor, np.array]:
  #Reshape the patterns tensor so that you can visualize all patterns at once as one square
  #That is, we want a heatmap that is filled with tiles of heatmaps (like the one right above)
  #The side length should be sqrt(num_examples) * sequence_length
  #The point of this is to get tensor manip practice, and get a birds eye view of the data

  #TODO
  sqrt_d = int(np.sqrt(patterns.shape[0]))

  new_patterns = []

  for i in range(sqrt_d):
    for j in range(sqrt_d):
      new_patterns.append(patterns[sqrt_d*i + j, :, :])

  # new_patterns = torch.stack(new_patterns)

  new_patterns = torch.stack(new_patterns).view(6, 6, 16, 16)  # Shape (6, 6, 16, 16)

  # Concatenate along the inner dimensions to form (96, 96)
  new_patterns = torch.cat([torch.cat([new_patterns[i, j] for j in range(6)], dim=1) for i in range(6)], dim=0)


  return new_patterns.cpu().numpy()

patterns = pattern[:, ind_head_idx]
sqrt_d = int(np.sqrt(len(dataset)))
print(patterns.shape)
patterns_all_data = reshape_heatmap(patterns)
px.imshow(patterns_all_data)

def visualize_all_attention_heads_side_by_side(cache, layer_idx):
    # Retrieve the attention patterns for the specified layer
    attention_patterns = cache["pattern", layer_idx, "attn"]
    num_heads = attention_patterns.shape[1]
    # Reshape and concatenate all head patterns
    all_head_patterns = []
    for head_idx in range(num_heads):
        head_pattern = attention_patterns[:, head_idx]
        all_head_patterns.append(head_pattern)
    # Stack and reshape to visualize all heads side by side
    all_head_patterns = torch.stack(all_head_patterns)
    sqrt_d = int(np.sqrt(all_head_patterns.shape[0]))
    new_patterns = []
    for i in range(sqrt_d):
        for j in range(sqrt_d):
            new_patterns.append(all_head_patterns[sqrt_d*i + j, :, :])
    new_patterns = torch.stack(new_patterns).view(sqrt_d, sqrt_d, *all_head_patterns.shape[1:])
    new_patterns = torch.cat([torch.cat([new_patterns[i, j] for j in range(sqrt_d)], dim=1) for i in range(sqrt_d)], dim=0)
    # Visualize the concatenated patterns
    px.imshow(new_patterns.cpu().numpy())

# Example usage
# visualize_all_attention_heads_side_by_side(ex_cache, layer_idx=3)