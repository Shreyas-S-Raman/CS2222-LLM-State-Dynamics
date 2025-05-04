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
import random
# same_action_diff_start = "Start at state {x1}. Take action {X1}, go to state {x3}. Take action {X3}, go to state {x2}. Take action {X1}, go to state {x4}. Take action {X2}, go to state {x1}. Take action {X1}, go to state"
# noop_distractors = []
from rich.progress import track
from functools import partial
import plotly.io as pio
import string
import re

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

    def update_sequence_generator(self, seq_gen):
        self.seq_generator = seq_gen

    def generate_and_classify_accuracy(self, num_samples:int=10, seq_len:int=10, detail=True, verbose=True):
        correct = 0
        correct_per_sample = []
        sequences = []
        label_tokens = []
        pred_tokens = []

        for _ in range(num_samples):
            sequence, label_token = self.seq_generator.generate(seq_len=seq_len)
            
            sequences.append(sequence)
            label_tokens.append(label_token)
            
            tokens = self.model.to_tokens(sequence, prepend_bos=True).to("cuda")
            logits, _ = self.model.run_with_cache(tokens)
            
            # probs = torch.nn.functional.softmax(logits, dim=-1)[0, -1, :]
            pred_token_id = torch.argmax(logits[0,-1,:], dim=-1)
            pred_token = self.model.to_string(pred_token_id)
            pred_tokens.append(pred_token)
            

            if 'dfa' in dir(self.seq_generator) and 'skip_actions' in dir(self.seq_generator):
                label_token_id = self.model.to_tokens(label_token, prepend_bos=False)
                label_token_id = torch.squeeze(label_token_id)
                if pred_token_id.item() == label_token_id.item():
                    correct += 1
                    correct_per_sample.append(True)
                else:
                    correct_per_sample.append(False)
            else:
                # Convert true_label to a set of potential true labels
                # Assuming label is a set of characters
                label_token_ids = [torch.squeeze(self.model.to_tokens(char, prepend_bos=False)).item() for char in label_token]

                # Squeeze each token tensor and combine them into a single tensor
                label_token_ids_set = set(label_token_ids)  # Assuming true_label is a tensor

                # Check if the predicted label is within the set of true labels
                if pred_token_id.item() in label_token_ids_set:
                    correct += 1
                    correct_per_sample.append(True)
                else:
                    correct_per_sample.append(False)
            
            
        if verbose:
            print('Sequence: ', sequence, 'Label:', label_token, 'Pred:', pred_token)
            if 'dfa' in dir(self.seq_generator):
                print(self.seq_generator.dfa)
        dfas = [self.seq_generator.dfa if 'dfa' in dir(self.seq_generator) else None for _ in range(num_samples)]
        return correct / num_samples, correct_per_sample, sequences, label_tokens, pred_tokens, dfas

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

    def normalize_patched_logit_diff(self, patched_logit_diff, corrupted_average_logit_diff, original_average_logit_diff) -> torch.Tensor: #a single item tensor (a number):
        # Subtract corrupted logit diff to measure the improvement, divide by the original - corrupted
        # 0 means zero change, negative means actively made worse, 1 means totally recovered clean performance, >1 means actively *improved* on clean performance

        return (patched_logit_diff - corrupted_average_logit_diff).item()/(original_average_logit_diff - corrupted_average_logit_diff).item()

    def patch_residual_component(self, 
        corrupted_residual_component,
        hook,
        pos_index: int, #the position in the sequence that you want to patch
        clean_cache: ActivationCache) -> torch.Tensor:
        #simply overwrite the corrupted_residual_component at the pos_index with that from the clean cache
        #hint: you can key the clean_cache with hook.name

        corrupted_residual_component[:, pos_index,:] = clean_cache[hook.name][:, pos_index,:]
        return corrupted_residual_component


    def create_patching_heatmap(self, regular_tokens, compressed_sequence, cache: ActivationCache, corrupted_tokens, answer_tokens, corrupted_average_logit_diff: float, original_average_logit_diff: float, filename: str):
        n_tokens = len(regular_tokens[0])

        n_layers = self.model.cfg.n_layers

        #Populate a heatmap of patched logit differences
        patched_positions = np.zeros((n_layers, n_tokens))

        for layer in track(list(range(n_layers))):
            for pos in range(n_tokens):
                hook_fn = partial(self.patch_residual_component, pos_index = pos, clean_cache = cache)
                patched_logits = self.model.run_with_hooks(
                    corrupted_tokens,
                    fwd_hooks = [(utils.get_act_name('resid_pre', layer), hook_fn)],
                    return_type='logits'
                )
                diff = ((patched_logits[0, -1, answer_tokens[0][0]]-patched_logits[0, -1, answer_tokens[0][1]]) + (patched_logits[1, -1, answer_tokens[1][0]]-patched_logits[1, -1, answer_tokens[1][1]]))/ 2    
                patched_positions[layer, pos] = self.normalize_patched_logit_diff(patched_logit_diff=diff, corrupted_average_logit_diff=corrupted_average_logit_diff, original_average_logit_diff=original_average_logit_diff)
        
        prompt_position_labels = [t for t in self.model.to_str_tokens(regular_tokens[0])]
        
        # fig = px.imshow(patched_positions, labels={'x':'Position', 'y':"Layer"}, x=prompt_position_labels, title=f"Logit difference for patching {compressed_sequence}")
        # # Update layout for smaller, tilted axis labels
        # fig.update_xaxes(tickangle=45, tickfont=dict(size=6))
        # fig.update_yaxes(tickfont=dict(size=8))
        # fig.write_image(filename)

        plt.figure(figsize=(max(10, len(prompt_position_labels) * 0.2), 3))  # Adjust width as needed
        im = plt.imshow(
            patched_positions,
            aspect='auto',
            cmap='plasma',
            vmin=0,
            vmax=1
        )

        plt.title(f"Logit difference for patching {compressed_sequence}")
        plt.xlabel("Position")
        plt.ylabel("Layer")

        plt.xticks(
            ticks=range(len(prompt_position_labels)),
            labels=prompt_position_labels,
            rotation=70,
            fontsize=4
        )
        plt.yticks(
            ticks=range(patched_positions.shape[0]),
            labels=[str(i) for i in range(patched_positions.shape[0])],
            fontsize=8
        )

        cbar = plt.colorbar(im, ticks=np.linspace(0, 1, 21))
        cbar.set_label('Logit Diff', fontsize=6)
        cbar.ax.tick_params(labelsize=6)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    def evaluate_noop_dfa(self, num_transitions: list[int], num_trials: int, num_saved_samples: int):
        #NOTE: large number of transitions e.g. 10, 20, 40, 50, 80, 100
        initial_transition = "Start at state {x1}. Take action {A1}, go to state {x2}. Take action {A2}, go to state {x1}."
        noop_buffer = "Take action {A}, go to state {x1}."
        final_action = "Take action {A1}, go to state" 

        mean_logit_diffs = []
        std_logit_diffs = []
        mean_accuracies = []

        for trans in num_transitions:
            mean_logit_diff = []
            saved_samples = num_saved_samples
            mean_accuracy = []
            for _ in range(num_trials):
                
                # Populate the set of actions and states used to make the DFA
                states = random.sample(string.ascii_lowercase, 2)
                actions = random.sample(string.ascii_uppercase, 2)
                noop_action_set = set(string.ascii_uppercase) - set(actions)
                
                #set all variables initially
                start_state = states[0]
                second_state = states[1]
                
                A1 = actions[0]
                A2 = actions[1]

                
                multi_noop_buffer = [noop_buffer.format(A=random.sample(noop_action_set,1)[0], x1=start_state)]*(trans-2)
                multi_noop_buffer = ' '.join(multi_noop_buffer)

                prompt = ' '.join([initial_transition.format(x1=start_state, A1=A1, A2=A2, x2=second_state), multi_noop_buffer, final_action.format(A1=A1)])
                counterfactual_prompt = ' '.join([initial_transition.format(x1=second_state, A1=A2, A2=A1, x2=start_state), multi_noop_buffer, final_action.format(A1=A2)])
                
                tokens = self.model.to_tokens([prompt, counterfactual_prompt], prepend_bos=True).to("cuda")
                logits, cache = self.model.run_with_cache(tokens, return_type="logits")


                answers = [(f" {second_state}", f" {start_state}"), (f" {start_state}", f" {second_state}")]
                answer_tokens = []
                for a in answers:
                    answer_tokens.append((self.model.to_single_token(a[0]), self.model.to_single_token(a[1]),))

                logit_diff = ((logits[0, -1, answer_tokens[0][0]]-logits[0, -1, answer_tokens[0][1]]) + (logits[1, -1, answer_tokens[1][0]]-logits[1, -1, answer_tokens[1][1]]))/ 2    
                
                if random.random() >= 0.5 and saved_samples > 0:
                    corrupted_tokens = self.model.to_tokens([counterfactual_prompt, prompt], prepend_bos=True).to("cuda")
                    corrupted_logits, __ = self.model.run_with_cache(corrupted_tokens, return_type="logits")

                    corrupted_logit_diff = ((corrupted_logits[0, -1, answer_tokens[0][0]]-corrupted_logits[0, -1, answer_tokens[0][1]]) + (corrupted_logits[1, -1, answer_tokens[1][0]]-corrupted_logits[1, -1, answer_tokens[1][1]]))/ 2    

                    compressed_sequence = ','.join(parse_states_and_actions(prompt=prompt))

                    self.create_patching_heatmap(regular_tokens=tokens, compressed_sequence=compressed_sequence, cache=cache, corrupted_tokens=corrupted_tokens, answer_tokens=answer_tokens, corrupted_average_logit_diff=corrupted_logit_diff, original_average_logit_diff=logit_diff, filename=os.path.join(Config.BASE_PATH, 'dfa_stateaction', f'noop_logitdiff_heatmap_{trans}trans.png'))


                    saved_samples -= 1


                mean_accuracy.append(torch.argmax(logits[0,-1,:])==answer_tokens[0][0])
                mean_accuracy.append(torch.argmax(logits[1,-1,:])==answer_tokens[1][0])
                mean_logit_diff.append(logit_diff.item())
            
            mean_accuracies.append(np.mean(mean_accuracy))
            mean_logit_diffs.append(np.mean(mean_logit_diff))
            std_logit_diffs.append(np.std(mean_logit_diff))
        
        '''
        PLOT 4: Plot the mean logit diffs chart as number of transitions increase
        '''
        pdb.set_trace()
        mean_logit_diffs = np.array(mean_logit_diffs)
        std_logit_diffs = np.array(std_logit_diffs)
        plt.plot(num_transitions, mean_logit_diffs, label='Mean Logit Diff')
        plt.fill_between(
            num_transitions,
            mean_logit_diffs - std_logit_diffs,
            mean_logit_diffs + std_logit_diffs,
            color='blue',
            alpha=0.2,
            label='Std Dev'
        )
        plt.xlabel('Number of Transitions')
        plt.ylabel('Mean Logit Differences')
        plt.title('Plot of Mean Logit Differences vs. Number of Transitions')
        # Save the figure
        plt.savefig(os.path.join(Config.BASE_PATH, 'dfa_stateaction', 'noop_logitdiff.png'))

        mean_accuracies = np.array(mean_accuracies)
        plt.plot(num_transitions, mean_accuracies, label='Mean Accuracy')
        plt.xlabel('Number of Transitions')
        plt.ylabel('Mean Accuracy')
        plt.title('Plot of Mean Accuracy vs. Number of Transitions')
        # Save the figure
        plt.savefig(os.path.join(Config.BASE_PATH, 'dfa_stateaction', 'noop_accuracy.png'))


        return mean_logit_diffs
    
    def evaluate_diff_state_same_action(self, num_transitions: list[int], num_trials: int, num_saved_samples: int):
        #NOTE: small number of transitions e.g. 6, 10, 14, 18, 22, 26, 30
        for t in num_transitions:
            assert t%4 == 2, "num_transitions has to give remainder 2 when div by 4"

        init_state = "Start at state {x1}."
        target_loop = "Take action {A1}, go to state {x3}. Take action {A2}, go to state {x1}."
        transition_to_distractor_loop = "Take action {A1}, go to state {x3}. Take action {A3}, go to state {x2}."
        distractor_loop = "Take action {A1}, go to state {x4}. Take action {A2}, go to state {x2}."
        transition_to_target_loop = "Take action {A1}, go to state {x4}. Take action {A3}, go to state {x1}."
        final_action = "Take action {A1}, go to state"

        mean_accuracies = []
        mean_logit_diffs = []
        std_logit_diffs = []

        for trans in num_transitions:
            mean_logit_diff = []
            mean_accuracy = []
            saved_samples = num_saved_samples
            for _ in range(num_trials):
                
                # Populate the set of actions and states used to make the DFA
                states = random.sample(string.ascii_lowercase, 4)
                actions = random.sample(string.ascii_uppercase, 3)
                
                #set all variables initially
                x1 = states[0]
                x2 = states[1]
                x3 = states[2]
                x4 = states[3]
                A1 = actions[0]
                A2 = actions[1]
                A3 = actions[2]

                prompt = [init_state.format(x1=x1)]
                counterfactual_prompt = [init_state.format(x1=x1)]
                for i in range(0, trans, 2):
                    counterfactual_prompt.append(target_loop.format(x1=x1, A1=A1, x3=x3, A2=A2))
                    
                    if (i//2) % 4 == 0:
                        #0, 8 -> 0, 4
                        prompt.append(target_loop.format(x1=x1, A1=A1, x3=x3, A2=A2))
                    if (i//2) % 4 == 1:
                        #2, 10 -> 1, 5
                        prompt.append(transition_to_distractor_loop.format(A3=A3, A1=A1, x3=x3, x2=x2))
                    if (i//2) % 4 == 2:
                        #4, 12 -> 2, 6
                        prompt.append(distractor_loop.format(x4=x4, A1=A1, x2=x2, A2=A2))
                    if (i//2) % 4 == 3:
                        #6, 14 -> 3, 7
                        prompt.append(transition_to_target_loop.format(A3=A3, A1=A1, x4=x4, x1=x1))

                prompt = ' '.join(prompt + [final_action.format(A1=A1)])
                counterfactual_prompt = ' '.join(counterfactual_prompt + [final_action.format(A1=A1)])

                tokens = self.model.to_tokens([prompt, counterfactual_prompt], prepend_bos=True).to("cuda")
                logits, cache = self.model.run_with_cache(tokens, return_type="logits")

                answers = [(f" {x4}", f" {x3}"), (f" {x3}", f" {x4}")]
                answer_tokens = []
                for a in answers:
                    answer_tokens.append((self.model.to_single_token(a[0]), self.model.to_single_token(a[1]),))
                logit_diff = ((logits[0, -1, answer_tokens[0][0]]-logits[0, -1, answer_tokens[0][1]]) + (logits[1, -1, answer_tokens[1][0]]-logits[1, -1, answer_tokens[1][1]]))/ 2    

                if random.random() >= 0.5 and saved_samples > 0:
                    corrupted_tokens = self.model.to_tokens([counterfactual_prompt, prompt], prepend_bos=True).to("cuda")
                    corrupted_logits, ___ = self.model.run_with_cache(corrupted_tokens, return_type="logits")

                    corrupted_logit_diff = ((corrupted_logits[0, -1, answer_tokens[0][0]]-corrupted_logits[0, -1, answer_tokens[0][1]]) + (corrupted_logits[1, -1, answer_tokens[1][0]]-corrupted_logits[1, -1, answer_tokens[1][1]]))/ 2    

                    compressed_sequence = ','.join(parse_states_and_actions(prompt=prompt))
                    self.create_patching_heatmap(regular_tokens = tokens, compressed_sequence=compressed_sequence, cache=cache, corrupted_tokens=corrupted_tokens, answer_tokens=answer_tokens, corrupted_average_logit_diff=corrupted_logit_diff, original_average_logit_diff=logit_diff, filename=os.path.join(Config.BASE_PATH, 'dfa_stateaction', f'diffstate_sameaction_logitdiff_heatmap_{trans}trans.png'))


                    saved_samples -= 1

                mean_accuracy.append(torch.argmax(logits[0,-1,:])==answer_tokens[0][0])
                mean_accuracy.append(torch.argmax(logits[1,-1,:])==answer_tokens[1][0])
                mean_logit_diff.append(logit_diff.item())
            mean_accuracies.append(np.mean(mean_accuracy))
            mean_logit_diffs.append(np.mean(mean_logit_diff))
            std_logit_diffs.append(np.std(mean_logit_diff))
        
        '''
        PLOT 4: Plot the mean logit diffs chart as number of transitions increase
        '''
        mean_logit_diff = np.array(mean_logit_diff)
        std_logit_diffs = np.array(std_logit_diffs)
        plt.plot(num_transitions, mean_logit_diffs, label='Mean Logit Diff')
        plt.fill_between(
            num_transitions,
            mean_logit_diffs - std_logit_diffs,
            mean_logit_diffs + std_logit_diffs,
            color='blue',
            alpha=0.2,
            label='Std Dev'
        )
        plt.xlabel('Number of Transitions')
        plt.ylabel('Mean Logit Differences')
        plt.title('Plot of Mean Logit Differences vs. Number of Transitions')
        # Save the figure
        plt.savefig(os.path.join(Config.BASE_PATH, 'dfa_stateaction', 'diff_state_same_action_logitdiff.png'))

        mean_accuracies = np.array(mean_accuracies)
        plt.plot(num_transitions, mean_accuracies, label='Mean Accuracy')
        plt.xlabel('Number of Transitions')
        plt.ylabel('Mean Accuracy')
        plt.title('Plot of Mean Accuracy vs. Number of Transitions')
        # Save the figure
        plt.savefig(os.path.join(Config.BASE_PATH, 'dfa_stateaction', 'diff_state_same_action_accuracy.png'))

        return mean_logit_diffs



def evaluate_dfa_stateaction_sequence(model_name:str, num_samples:int, init_states:list, init_transitions:list, max_states:int, min_states:int, state_interval:int, max_transitions:int, min_transitions:int, transition_interval:int, density_interval:int, reduce_states: bool=False):
    
    #tracking correct and incorrect outputs
    correct_output = {}
    incorrect_output = {}
    dfa_output = {}
    #track 3D heatmaps of accuracy for DFA state action environment
    accuracy_heatmap = np.zeros((len(init_states)+(max_states-min_states)//state_interval, len(init_transitions) + (max_transitions-min_transitions)//transition_interval, density_interval))

    eval_pipeline = EvaluationPipeline(model_name=model_name, sequence_generator=None)

    pdb.set_trace()
    eval_pipeline.evaluate_noop_dfa(num_transitions=Config.noop_transitions, num_trials=Config.noop_trials, num_saved_samples=1)
    eval_pipeline.evaluate_diff_state_same_action(num_transitions=Config.diff_state_same_action_transitions, num_trials=Config.diff_state_same_action_trials, num_saved_samples=1)
    
    

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
                
                eval_pipeline.update_sequence_generator(seq_generator)
                acc, correct_per_sample, sequences, labels, pred_label, dfas = eval_pipeline.generate_and_classify_accuracy(num_samples=num_samples, seq_len=trans)
                
                for (correct,s,l, p, d) in zip(correct_per_sample, sequences, labels, pred_label, dfas):
                    
                    if correct:
                        correct_output[state][trans][density].append([s, l, p])
                    else:
                        incorrect_output[state][trans][density].append([s, l, p])
                    dfa_output[state][trans][density].append(d)
                accuracy_heatmap[i][j][k] = acc
                print(f"STATE {state} | TRANS {trans} | DENSITY {density} | ACCURACY: {acc*100}%")
    
    generate_all_accuracy_plots(accuracy_heatmap=accuracy_heatmap, init_transitions=init_transitions, min_transitions=min_transitions, max_transitions=max_transitions, transition_interval=transition_interval, init_states=init_states, min_states=min_states, max_states=max_states, state_interval=state_interval, dfa_type='dfa_stateaction')

    

    #save all the outputs and heatmaps
    with open(os.path.join(Config.BASE_PATH, 'dfa_stateaction', 'correct_dfa_stateaction.json'), 'w') as f:
        json.dump(correct_output, f, indent=4)
    with open(os.path.join(Config.BASE_PATH, 'dfa_stateaction', 'incorrect_dfa_stateaction.json'), 'w') as f:
        json.dump(incorrect_output, f, indent=4)
    with open(os.path.join(Config.BASE_PATH, 'dfa_stateaction', 'dfa_stateaction.pkl'), 'wb') as f:
        pickle.dump(dfa_output, f)
   

def evaluate_dfa_statestate_sequence(model_name:str, num_samples:int, init_states:list, init_transitions:list, max_states:int, min_states:int, state_interval:int, max_transitions:int, min_transitions:int, transition_interval:int, density_interval:int, reduce_states: bool=False):
    #tracking correct and incorrect outputs
    correct_output = {}
    incorrect_output = {}
    dfa_output = {}
    #track 3D heatmaps of accuracy for DFA state action environment
    accuracy_heatmap = np.zeros((len(init_states)+(max_states-min_states)//state_interval, len(init_transitions) + (max_transitions-min_transitions)//transition_interval, density_interval))
    eval_pipeline = EvaluationPipeline(model_name=model_name, sequence_generator=None)

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
                eval_pipeline.update_sequence_generator(seq_generator)
                acc, correct_per_sample, sequences, labels, pred_label, dfas = eval_pipeline.generate_and_classify_accuracy(num_samples=num_samples, seq_len=trans)
                
                for (correct,s,l,p,d) in zip(correct_per_sample, sequences, labels, pred_label, dfas):
                    if correct:
                        correct_output[state][trans][density].append([s, list(l), p])
                    else:
                        incorrect_output[state][trans][density].append([s, list(l), p])
                    dfa_output[state][trans][density].append(d)
                accuracy_heatmap[i][j][k] = acc
                print(f"STATE {state} | TRANS {trans} | DENSITY {density} | ACCURACY: {acc*100}%")
    
    generate_all_accuracy_plots(accuracy_heatmap=accuracy_heatmap, init_transitions=init_transitions, min_transitions=min_transitions, max_transitions=max_transitions, transition_interval=transition_interval, init_states=init_states, min_states=min_states, max_states=max_states, state_interval=state_interval, dfa_type='dfa_statestate')
    
    
    #save all the outputs and heatmaps
    with open(os.path.join(Config.BASE_PATH, 'dfa_statestate', 'correct_dfa_statestate.json'), 'w') as f:
        json.dump(correct_output, f, indent=4)
    with open(os.path.join(Config.BASE_PATH, 'dfa_statestate', 'incorrect_dfa_statestate.json'), 'w') as f:
        json.dump(incorrect_output, f, indent=4)
    with open(os.path.join(Config.BASE_PATH, 'dfa_statestate', 'dfa_statestate.pkl'), 'wb') as f:
        pickle.dump(dfa_output, f)
   

def generate_all_accuracy_plots(accuracy_heatmap, init_transitions, min_transitions, max_transitions, transition_interval, init_states, min_states, max_states, state_interval, dfa_type):
    os.makedirs(os.path.join(Config.BASE_PATH, dfa_type), exist_ok=True)
    '''PLOT 1: General Accuracy'''
    # Get the shape of the heatmap
    num_states, num_transitions, density_intervals = accuracy_heatmap.shape

    # Create a figure for each density interval
    for i, density in  enumerate([0.0, 0.5, 1.0]):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Extract the 2D slice for the current density interval
        accuracy_slice = accuracy_heatmap[:, :, i].T
        
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
        ax.set_ylabel("Number of Transitions")
        ax.set_xlabel("Number of State")
        

        # Annotate each cell with the accuracy value
        for y in range(accuracy_slice.shape[0]):
            for x in range(accuracy_slice.shape[1]):
                ax.text(x, y, f"{accuracy_slice[y, x]:.2f}", ha='center', va='center', color='black', fontsize=5)
        
        #update plot tick labels based on the range of states and transitions
        ax.set_yticks(np.arange(len(init_transitions + list(range(min_transitions, max_transitions, transition_interval)))))
        ax.set_yticklabels(init_transitions + list(range(min_transitions, max_transitions, transition_interval)))
        ax.set_xticks(np.arange(len(init_states + list(range(min_states, max_states, state_interval)))))
        ax.set_xticklabels(init_states + list(range(min_states, max_states, state_interval)))
        
        # Save the plot
        path = f'./{Config.model_name}' if not Config.reduce_states else f'./{Config.model_name}_reduced'
        Config.BASE_PATH = os.path.relpath(path)
        plt.savefig(os.path.join(Config.BASE_PATH, dfa_type, f"accuracy_{dfa_type}_density_{density}.png"), dpi=300)
        plt.close()

    '''PLOT 2: Difference from random chance'''
    # Get the shape of the heatmap for diff from random
    num_states, num_transitions, density_intervals = accuracy_heatmap.shape

    # Create a figure for each density interval
    for i, density in  enumerate([0.0, 0.5, 1.0]):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Extract the 2D slice for the current density interval
        accuracy_diff_slice = accuracy_heatmap[:, :, i] - np.array([1/(i+1) for i in range(len(accuracy_heatmap.shape[0]))])
        accuracy_diff_slice = accuracy_diff_slice.T
        
        # Plot the heatmap
        cax = ax.imshow(accuracy_diff_slice, cmap=viridis, norm=norm, origin='lower')
        
        # Add color bar
        cbar = fig.colorbar(cax, ax=ax, pad=0.1)
        cbar.set_label('Accuracy Diff')
        
        # Set labels and title
        ax.set_title(f"Density Interval {density}")
        ax.set_xlabel("Number of Transitions")
        ax.set_ylabel("Number of State")
        

        # Annotate each cell with the accuracy value
        for y in range(accuracy_diff_slice.shape[0]):
            for x in range(accuracy_diff_slice.shape[1]):
                ax.text(x, y, f"{accuracy_diff_slice[y, x]:.2f}", ha='center', va='center', color='black', fontsize=5)
        
        #update plot tick labels based on the range of states and transitions
        ax.set_xticks(np.arange(len(init_transitions + list(range(min_transitions, max_transitions, transition_interval)))))
        ax.set_xticklabels(init_transitions + list(range(min_transitions, max_transitions, transition_interval)))
        ax.set_yticks(np.arange(len(init_states + list(range(min_states, max_states, state_interval)))))
        ax.set_yticklabels(init_states + list(range(min_states, max_states, state_interval)))
        
        # Save the plot
        path = f'./{Config.model_name}' if not Config.reduce_states else f'./{Config.model_name}_reduced'
        Config.BASE_PATH = os.path.relpath(path)
        plt.savefig(os.path.join(Config.BASE_PATH, dfa_type, f"accuracy_{dfa_type}_density_{density}_diff.png"), dpi=300)
        plt.close()

    np.savez(os.path.join(Config.BASE_PATH, dfa_type, f'accuracy_{dfa_type}.npz'), array=accuracy_heatmap)
    
    '''PLOT 3: Line graph vs random chance'''
    accuracy_array = accuracy_array[:,:,1]*100

    # Plot random baseline
    baseline = [100 / n for n in init_states + list(range(min_states, max_states, state_interval))]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(init_states + list(range(min_states, max_states, state_interval)), baseline, '--', color='gray', label='Random Baseline', alpha=0.7)
    
    # Color map for transition curves
    colors = plt.cm.viridis(np.linspace(0, 1, len(init_transitions + list(range(min_transitions, max_transitions, transition_interval)))))
    
    # Plot accuracy curves for each transition count
    for i, num_transitions in enumerate(init_transitions + list(range(min_transitions, max_transitions, transition_interval))):
        if i%2 != 0:
            continue
        ax.plot(
            init_states + list(range(min_states, max_states, state_interval)), 
            accuracy_array[:,i], 
            'o-', 
            color=colors[i], 
            label=f'{num_transitions} Transition{"s" if num_transitions > 1 else ""}'
        )
    
    ax.set_xticks(init_states + list(range(min_states, max_states, state_interval)))
    ax.set_xlabel('# states', fontsize=20)
    ax.set_ylabel('acc (%)', fontsize=20)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    
    ax.legend(loc='best', fontsize=13)
    
    plt.tight_layout()
    path = f'./{Config.model_name}' if not Config.reduce_states else f'./{Config.model_name}_reduced'
    Config.BASE_PATH = os.path.relpath(path)
    plt.savefig(os.path.join(Config.BASE_PATH, dfa_type, f"accuracy_{dfa_type}_lineplot.png"), dpi=300)
    plt.close()

    np.savez(os.path.join(Config.BASE_PATH, dfa_type, f'accuracy_{dfa_type}.npz'), array=accuracy_heatmap)

def evaluate_random_sequence(model_name:str, num_samples:int, init_states:list, init_transitions:list, max_states:int, min_states:int, state_interval:int, max_transitions:int, min_transitions:int, transition_interval:int):
    #tracking correct and incorrect outputs
    correct_output = {}
    incorrect_output = {}
    #track 2D heatmap of accuracy
    accuracy_heatmap = np.zeros((len(init_states)+(max_states-min_states)//state_interval, len(init_transitions) + (max_transitions-min_transitions)//transition_interval))
    eval_pipeline = EvaluationPipeline(model_name=model_name, sequence_generator=None)

    for i, state in tqdm(enumerate(init_states + list(range(min_states, max_states, state_interval)))):
        for j, trans in enumerate(init_transitions + list(range(min_transitions, max_transitions, transition_interval))):
            
            seq_generator = RandomLetterSequenceGenerator(length=trans, repeat_pattern_len=state)
            eval_pipeline.update_sequence_generator(seq_generator)
            
            acc, correct_per_sample, sequences, labels, __, __ = eval_pipeline.generate_and_classify_accuracy(num_samples=num_samples, seq_len=trans)

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
            
    os.makedirs(os.path.join(Config.BASE_PATH, 'random'), exist_ok=True)
    #save all the outputs and heatmaps
    with open(os.path.join(Config.BASE_PATH, 'random', 'correct_random_letter.json'), 'w') as f:
        json.dump(correct_output, f, indent=4)
    with open(os.path.join(Config.BASE_PATH, 'random', 'incorrect_random_letter.json'), 'w') as f:
        json.dump(incorrect_output, f, indent=4)
    
    # Optional: label the rows and columns
    state_labels = init_states + [state for state in range(min_states, max_states, state_interval)]
    trans_labels = init_transitions + [trans for trans in range(min_transitions, max_transitions, transition_interval)]
    df = pd.DataFrame(accuracy_heatmap, index=state_labels, columns=trans_labels).T

    # Plot and save heatmap
    plt.figure(figsize=(8, 8))
    ax = sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", annot_kws={"size": 5, "rotation": 30})# Label colorbar
    colorbar = ax.collections[0].colorbar
    colorbar.set_label('Accuracy')
    plt.title(f"Random Sequence | {model_name}")
    plt.xlabel("Number of States")
    plt.ylabel("Number of Transitions")
    plt.savefig(os.path.join(Config.BASE_PATH,'random', "accuracy_random_letter.png"), dpi=300)
    plt.close()

    np.savez(os.path.join(Config.BASE_PATH, 'random', 'accuracy_random_letter.npz'), array=accuracy_heatmap)

def parse_states_and_actions(prompt: str):
    # Regular expression patterns to match states and actions
    pattern = r'state (\w+)|action (\w+)'  # Matches 'state x1', 'action A1', etc.

    # Find all matches for states and actions
    matches = re.findall(pattern, prompt)

    # Flatten the list of tuples and filter out empty strings
    ordered_elements = [item for sublist in matches for item in sublist if item]

    return ordered_elements

# Example usage
# prompt = "Start at state x1. Take action A1, go to state x2. Take action A2, go to state x1."
# ordered_elements = parse_states_and_actions(prompt)
# print("Ordered Elements:", ordered_elements)

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


   
















            
            

            
