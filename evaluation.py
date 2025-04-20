import torch
import numpy as np
from transformer_lens import HookedTransformer, HookedTransformerConfig, ActivationCache
import transformer_lens.utils as utils
import circuitsvis as cv
from config import Config
import plotly.express as px
from sequence_generator import SequenceGenerator,RandomLetterSequenceGenerator, DFAStateActionSequenceGenerator, DFAStateSequenceGenerator
import json
import pickle
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, viridis
import pdb 
from tqdm import tqdm

class EvaluationPipeline:
    def __init__(self, model_name:str, sequence_generator):
        self.model = HookedTransformer.from_pretrained(
            model_name,
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=True,
            low_cpu_mem_usage=True
        )
        self.seq_generator = sequence_generator

    def generate_and_classify_accuracy(self, num_samples:int=10, seq_len:int=10, detail=True, verbose=True):
        correct = 0
        correct_per_sample = []
        sequences = []
        labels = []

        for _ in range(num_samples):
            sequence, label = self.seq_generator.generate(seq_len=seq_len)
            if detail:
                sequences.append(sequence)
                labels.append(label)
            
            tokens = self.model.to_tokens(sequence, prepend_bos=True)
            logits, _ = self.model.run_with_cache(tokens)
            
            probs = torch.nn.functional.softmax(logits, dim=-1)[0, -1, :]
            pred_label = torch.argmax(probs, axis=-1)
            next_token = self.model.to_string(pred_label)

            true_label = self.model.to_tokens(label, prepend_bos=False)
            true_label = torch.squeeze(true_label)
            if pred_label == true_label:
                correct += 1
                correct_per_sample.append(True)
            else:
                correct_per_sample.append(False)
        if verbose:
            print('Sequence: ', sequence, 'Label: ', label, 'Pred: ', next_token)
        dfas = [self.seq_generator.dfa if 'dfa' in dir(self.seq_generator) else None for _ in range(num_samples)]
        return correct / num_samples, correct_per_sample, sequences, labels, dfas

    def get_token_probabilities(self, seq_len:int):
        # Return probabilities of likely next tokens
        sequence, label = self.seq_generator.generate(seq_len=seq_len)
        tokens = self.model.to_tokens(sequence, prepend_bos=True)
        logits, _ = self.model.run_with_cache(tokens)
        probs = torch.nn.functional.softmax(logits, dim=-1)[0, -1, :]
        pred_label = torch.argmax(probs, axis=-1)
        next_token_str = self.model.to_string(pred_label)

        return probs, next_token_str

    def get_activation_patterns(self, input_sequence:str, layer_idx:int, seq_len:int):
        # Output activation/attention patterns for specified layer and head
        sequence, label = self.seq_generator.generate(seq_len=seq_len)
        tokens = self.model.to_tokens(sequence, prepend_bos=True)
        _, cache = self.model.run_with_cache(tokens)
        str_tokens = self.model.to_str_tokens(input_sequence, prepend_bos=False)
        attention_pattern = cache['pattern', layer_idx, 'attn']
        print(attention_pattern.shape)
        print(f"Layer {layer_idx} Head Attention Patterns:")
        cv.attention.attention_patterns(tokens=str_tokens, attention=attention_pattern)

    def visualize_all_attention_heads(self, layer_idx:int, seq_len:int):
        # Retrieve the attention patterns for the specified layer
        sequence, label = self.seq_generator.generate(seq_len=seq_len)
        tokens = self.model.to_tokens(sequence, prepend_bos=True)
        _, cache = self.model.run_with_cache(tokens)
        str_tokens = self.model.to_str_tokens(sequence, prepend_bos=False)
        attention_patterns = cache["pattern", layer_idx, "attn"]
        
        # Number of attention heads
        num_heads = attention_patterns.shape[1]
        
        # Iterate over each head and visualize the attention pattern
        for head_idx in range(num_heads):
            head_pattern = attention_patterns[:, head_idx]
            print(f"Layer {layer_idx} Head {head_idx} Attention Pattern:")
            cv.attention.attention_patterns(tokens=str_tokens, attention=head_pattern)

    def visualize_all_attention_heads_side_by_side(self, seq_len:int, layer_idx:int):
       # Retrieve the attention patterns for the specified layer
        sequence, label = self.seq_generator.generate(seq_len=seq_len)
        tokens = self.model.to_tokens(sequence, prepend_bos=True)
        _, cache = self.model.run_with_cache(tokens)
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

    def get_logits_and_logprobs(self, input_sequence:str, label:str):
        utils.test_prompt(input_sequence, label, self.model, prepend_bos=True)
        # Output logits and log probabilities of different tokens
        tokens = self.model.to_tokens(input_sequence, prepend_bos=True)
        logits, _ = self.model.run_with_cache(tokens)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return logits, log_probs

def evaluate_dfa_stateaction_sequence(model_name:str, num_samples:int, init_states:list, init_transitions:list, max_states:int, min_states:int, state_interval:int, max_transitions:int, min_transitions:int, transition_interval:int, density_interval:int, reduce_states: bool=False):
     #tracking correct and incorrect outputs
    correct_output = {}
    incorrect_output = {}
    dfa_output = {}
    #track 3D heatmaps of accuracy for DFA state action environment
    accuracy_heatmap = np.zeros((len(init_states)+(max_states-min_states)//state_interval, len(init_transitions) + (max_transitions-min_transitions)//transition_interval, density_interval))

    for i, state in tqdm(enumerate(init_states + list(range(min_states, max_states, state_interval)))):
        for j, trans in enumerate(init_transitions + list(range(min_transitions, max_transitions, transition_interval))):
                
            max_density = max(1, state*(state-1)//2 + 1)
            
            for k, density in enumerate([state, int((state+max_density)//2), max_density]):
                if state not in correct_output:
                    correct_output[state] = {}
                if trans not in correct_output[state]:
                    correct_output[state][trans] = {}
                if density not in correct_output[state][trans]:
                    correct_output[state][trans][density] = []
                if state not in incorrect_output:
                    incorrect_output[state] = {}
                if trans not in incorrect_output[state]:
                    incorrect_output[state][trans] = {}
                if density not in incorrect_output[state][trans]:
                    incorrect_output[state][trans][density] = []
                if state not in dfa_output:
                    dfa_output[state] = {}
                if trans not in dfa_output[state]:
                    dfa_output[state][trans] = {}
                if density not in dfa_output[state][trans]:
                    dfa_output[state][trans][density] = []
                
                seq_generator = DFAStateActionSequenceGenerator(
                                num_states=state, 
                                num_edges=density,
                                num_unique_actions=400 if not reduce_states else 26,
                                max_sink_nodes=1,
                                reduce_states = reduce_states
                                )
                eval_pipeline = EvaluationPipeline(model_name=model_name, sequence_generator=seq_generator)
                
                acc, correct_per_sample, sequences, labels, dfas = eval_pipeline.generate_and_classify_accuracy(num_samples=num_samples, seq_len=trans)
                
                for (correct,s,l, d) in zip(correct_per_sample, sequences, labels, dfas):
                    
                    if correct:
                        correct_output[state][trans][density].append([s, l])
                    else:
                        incorrect_output[state][trans][density].append([s, l])
                    dfa_output[state][trans][density].append(d)
                accuracy_heatmap[i][j][k] = acc
                print(f"STATE {state} | TRANS {trans} | DENSITY {density} | ACCURACY: {acc*100}%")
                
    
    #save all the outputs and heatmaps
    os.mkdir(os.path.join(Config.BASE_PATH, 'dfa_stateaction'))
    with open(os.path.join(Config.BASE_PATH, 'dfa_stateaction', 'correct_dfa_stateaction.json'), 'w') as f:
        json.dump(correct_output, f, indent=4)
    with open(os.path.join(Config.BASE_PATH, 'dfa_stateaction', 'incorrect_dfa_stateaction.json'), 'w') as f:
        json.dump(incorrect_output, f, indent=4)
    with open(os.path.join(Config.BASE_PATH, 'dfa_stateaction', 'dfa_stateaction.json'), 'w') as f:
        pickle.dump(dfa_output, f)
   
    # Get the shape of the heatmap
    num_states, num_transitions, density_intervals = accuracy_heatmap.shape

    # Create a figure for each density interval
    for i, density in  enumerate([0.0, 0.5, 1.0]):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Extract the 2D slice for the current density interval
        accuracy_slice = accuracy_heatmap[:, :, i]
        
        # Normalize the accuracy values for color mapping
        norm = Normalize(vmin=accuracy_slice.min(), vmax=accuracy_slice.max())
        colors = viridis(norm(accuracy_slice))
        
        # Plot the heatmap
        cax = ax.imshow(accuracy_slice, cmap=viridis, norm=norm, origin='lower')
        
        # Add color bar
        cbar = fig.colorbar(cax, ax=ax, pad=0.1)
        cbar.set_label('Accuracy')
        
        # Set labels and title
        ax.set_title(f"Density Interval {density}")
        ax.set_xlabel("Number of States")
        ax.set_ylabel("Number of Transitions")
        
        # Set ticks
        ax.set_xticks(np.arange(num_states))
        ax.set_yticks(np.arange(num_transitions))
        ax.set_xticklabels(np.arange(1, num_states + 1))
        ax.set_yticklabels(np.arange(1, num_transitions + 1))
        
        # Save the plot
        plt.savefig(os.path.join(Config.BASE_PATH, 'dfa_stateaction', f"accuracy_dfastateaction_density_{density}.png"), dpi=300)
        plt.close()

    np.savez(os.path.join(Config.BASE_PATH, 'dfa_stateaction', 'accuracy_dfastateaction.npz'), array=accuracy_heatmap)

def evaluate_dfa_statestate_sequence(model_name:str, num_samples:int, init_states:list, init_transitions:list, max_states:int, min_states:int, state_interval:int, max_transitions:int, min_transitions:int, transition_interval:int, density_interval:int, reduce_states: bool=False):
    #tracking correct and incorrect outputs
    correct_output = {}
    incorrect_output = {}
    dfa_output = {}
    #track 3D heatmaps of accuracy for DFA state action environment
    accuracy_heatmap = np.zeros((len(init_states)+(max_states-min_states)//state_interval, len(init_transitions) + (max_transitions-min_transitions)//transition_interval, density_interval))

    for i, state in tqdm(enumerate(init_states + list(range(min_states, max_states, state_interval)))):
        for j, trans in enumerate(init_transitions + list(range(min_transitions, max_transitions, transition_interval))):
                
            max_density = max(1, state*(state-1)//2 + 1)
            
            for k, density in enumerate([state, int((state+max_density)//2), max_density]):
                if state not in correct_output:
                    correct_output[state] = {}
                if trans not in correct_output[state]:
                    correct_output[state][trans] = {}
                if density not in correct_output[state][trans]:
                    correct_output[state][trans][density] = []
                if state not in incorrect_output:
                    incorrect_output[state] = {}
                if trans not in incorrect_output[state]:
                    incorrect_output[state][trans] = {}
                if density not in incorrect_output[state][trans]:
                    incorrect_output[state][trans][density] = []
                if state not in dfa_output:
                    dfa_output[state] = {}
                if trans not in dfa_output[state]:
                    dfa_output[state][trans] = {}
                if density not in dfa_output[state][trans]:
                    dfa_output[state][trans][density] = []
                
                seq_generator = DFAStateSequenceGenerator(
                                num_states=state, 
                                num_edges=density,
                                max_sink_nodes=1,
                                reduce_states = reduce_states
                                )
                eval_pipeline = EvaluationPipeline(model_name=model_name, sequence_generator=seq_generator)
                
                acc, correct_per_sample, sequences, labels, dfas = eval_pipeline.generate_and_classify_accuracy(num_samples=num_samples, seq_len=trans)
                
                for (correct,s,l, d) in zip(correct_per_sample, sequences, labels, dfas):
                    if correct:
                        correct_output[state][trans][density].append([s, l])
                    else:
                        incorrect_output[state][trans][density].append([s, l])
                    dfa_output[state][trans][density].append(d)
                accuracy_heatmap[i][j][k] = acc
                print(f"STATE {state} | TRANS {trans} | DENSITY {density} | ACCURACY: {acc*100}%")
                
    
    #save all the outputs and heatmaps
    os.mkdir(os.path.join(Config.BASE_PATH, 'dfa_statestate'))
    with open(os.path.join(Config.BASE_PATH, 'dfa_statestate', 'correct_dfa_statestate.json'), 'w') as f:
        json.dump(correct_output, f, indent=4)
    with open(os.path.join(Config.BASE_PATH, 'dfa_statestate', 'incorrect_dfa_statestate.json'), 'w') as f:
        json.dump(incorrect_output, f, indent=4)
    with open(os.path.join(Config.BASE_PATH, 'dfa_statestate', 'dfa_statestate.pkl'), 'w') as f:
        pickle.dump(dfa_output, f)
   
    # Get the shape of the heatmap
    num_states, num_transitions, density_intervals = accuracy_heatmap.shape

    # Create a figure for each density interval
    for i, density in  enumerate([0.0, 0.5, 1.0]):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Extract the 2D slice for the current density interval
        accuracy_slice = accuracy_heatmap[:, :, i]
        
        # Normalize the accuracy values for color mapping
        norm = Normalize(vmin=accuracy_slice.min(), vmax=accuracy_slice.max())
        colors = viridis(norm(accuracy_slice))
        
        # Plot the heatmap
        cax = ax.imshow(accuracy_slice, cmap=viridis, norm=norm, origin='lower')
        
        # Add color bar
        cbar = fig.colorbar(cax, ax=ax, pad=0.1)
        cbar.set_label('Accuracy')
        
        # Set labels and title
        ax.set_title(f"Density Interval {density}")
        ax.set_xlabel("Number of States")
        ax.set_ylabel("Number of Transitions")
        
        # Set ticks
        ax.set_xticks(np.arange(num_states))
        ax.set_yticks(np.arange(num_transitions))
        ax.set_xticklabels(np.arange(1, num_states + 1))
        ax.set_yticklabels(np.arange(1, num_transitions + 1))
        
        # Save the plot
        plt.savefig(os.path.join(Config.BASE_PATH, 'dfa_statestate', f"accuracy_dfa_statestate_density_{density}.png"), dpi=300)
        plt.close()

    np.savez(os.path.join(Config.BASE_PATH, 'dfa_statestate', 'accuracy_dfastatestate.npz'), array=accuracy_heatmap)

def evaluate_random_sequence(model_name:str, num_samples:int, init_states:list, init_transitions:list, max_states:int, min_states:int, state_interval:int, max_transitions:int, min_transitions:int, transition_interval:int):
    #tracking correct and incorrect outputs
    correct_output = {}
    incorrect_output = {}
    #track 2D heatmap of accuracy
    accuracy_heatmap = np.zeros((len(init_states)+(max_states-min_states)//state_interval, len(init_transitions) + (max_transitions-min_transitions)//transition_interval))

    for i, state in tqdm(enumerate(init_states + list(range(min_states, max_states, state_interval)))):
        for j, trans in enumerate(init_transitions + list(range(min_transitions, max_transitions, transition_interval))):
            
            seq_generator = RandomLetterSequenceGenerator(length=trans, repeat_pattern_len=state)
            eval_pipeline = EvaluationPipeline(model_name=model_name, sequence_generator=seq_generator)

            acc, correct_per_sample, sequences, labels, __ = eval_pipeline.generate_and_classify_accuracy(num_samples=num_samples, seq_len=trans)

            for (correct,s,l) in zip(correct_per_sample, sequences, labels):
                if state not in correct_output:
                    correct_output[state] = {}
                if trans not in correct_output[state]:
                    correct_output[state][trans] = []
               
                if state not in incorrect_output:
                    incorrect_output[state] = {}
                if trans not in incorrect_output[state]:
                    incorrect_output[state][trans] = []
                
                if correct:
                    correct_output[state][trans].append([s, l])
                else:
                    incorrect_output[state][trans].append([s, l])
            
            accuracy_heatmap[i][j] = acc
            print(f"STATE {state} | TRANS {trans} | ACCURACY: {acc*100}%")
            
    os.mkdir(os.path.join(Config.BASE_PATH, 'random'))
    #save all the outputs and heatmaps
    with open(os.path.join(Config.BASE_PATH, 'random', 'correct_random_letter.json'), 'w') as f:
        json.dump(correct_output, f, indent=4)
    with open(os.path.join(Config.BASE_PATH, 'random', 'incorrect_random_letter.json'), 'w') as f:
        json.dump(incorrect_output, f, indent=4)
    
    # Optional: label the rows and columns
    row_labels = init_states + [state for state in range(min_states, max_states, state_interval)]
    col_labels = init_transitions + [trans for trans in range(min_transitions, max_transitions, transition_interval)]
    df = pd.DataFrame(accuracy_heatmap, index=row_labels, columns=col_labels)

    # Plot and save heatmap
    plt.figure(figsize=(8, 8))
    ax = sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", annot_kws={"size": 5, "rotation": 30})# Label colorbar
    colorbar = ax.collections[0].colorbar
    colorbar.set_label('Accuracy')
    plt.title(f"Random Sequence | {model_name}")
    plt.xlabel("Number of Transitions")
    plt.ylabel("Number of States")
    plt.savefig(os.path.join(Config.BASE_PATH,'random', "accuracy_random_letter.png"), dpi=300)
    plt.close()

    np.savez(os.path.join(Config.BASE_PATH, 'random', 'accuracy_random_letter.npz'), array=accuracy_heatmap)

if __name__ == '__main__':
    pdb.set_trace()

    
    '''
    https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html?utm_source=chatgpt.com
    Range of values to try:
    GPT2: gpt2-small, gpt2-medium, gpt2-large, gpt2-xl, othello-gpt
    OPT: facebook/opt-125m, facebook/opt-2.7b, facebook/opt-6.7b
    TinyStories: "tiny-stories-1M", "tiny-stories-3M", tiny-stories-28M
    Pythia: pythia-14m, pythia-70m, pythia-1.4b
    LLaMa: llama-7b, llama-13b, llama-30b [DOESN'T WORK DIRECTLY]
    T5: t5-small, t5-base, t5-large
    '''


   
















            
            

            
