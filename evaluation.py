import torch
import numpy as np
from transformer_lens import HookedTransformer, HookedTransformerConfig, ActivationCache
import transformer_lens.utils as utils
import circuitsvis as cv
from config import Config
import plotly.express as px
from sequence_generator import RandomLetterSequenceGenerator, DFAStateActionSequenceGenerator, DFAStateSequenceGenerator, DFAPDDLSequenceGenerator
from sequence_generator import factorial
import json
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, viridis
import pdb 

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

    def generate_and_classify_accuracy(self, num_samples:int=10, seq_len:int=10, detail=True, verbose=False):
        # Generate sequences and classify accuracy
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
            if verbose:
                print('Sequence: ', sequence, 'Label: ', label, 'Pred: ', next_token)

            true_label = self.model.to_tokens(label, prepend_bos=False)
            true_label = torch.squeeze(true_label)
            if pred_label == true_label:
                correct += 1
                correct_per_sample.append(True)
            else:
                correct_per_sample.append(False)
        
        return correct / num_samples, correct_per_sample, sequences, labels

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
        """
        Visualize the attention headmap for all attention heads in a given layer.

        :param cache: The activation cache containing attention patterns.
        :param layer_idx: The index of the layer to visualize.
        :param tokens: The input tokens for which the attention patterns were computed.
        """
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

if __name__ == '__main__':
    pdb.set_trace()

    #num states range
    init_states = [1,2,3,4,5,6,7,8,9]
    min_states = 10
    max_states = 110
    state_interval = 10
    #transition range
    init_transitions = [1,2,3,4,5,6,7,8,9]
    min_transitions = 10
    max_transitions = 110
    transition_interval = 10
    #density interval
    density_interval = 10


    #range of samples
    num_samples = 50

    #name of model
    model_name = 'pythia-70m'
    '''
    https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html?utm_source=chatgpt.com
    Range of values to try:
    GPT2: gpt2-small, gpt2-medium, gpt2-large, gpt2-xl
    OPT: facebook/opt-125m, facebook/opt-2.7b, facebook/opt-13b
    TinyStories: "tiny-stories-1M", "tiny-stories-3M", tiny-stories-28M
    Pythia: pythia-14m, pythia-70m, pythia-1.4b
    LLaMa: llama-7b, llama-13b, llama-30b
    T5: t5-small, t5-base, t5-large
    '''

    BASE_PATH = os.path.relpath(f'./{model_name}')
    if not os.path.exists(BASE_PATH):
        os.mkdir(BASE_PATH)

    # #tracking correct and incorrect outputs
    # correct_output = {}
    # incorrect_output = {}
    # #track 2D heatmap of accuracy
    # accuracy_heatmap = np.zeros((len(init_states)+(max_states-min_states)//state_interval, len(init_transitions) + (max_transitions-min_transitions)//transition_interval))

    # for i, state in enumerate(init_states + list(range(min_states, max_states, state_interval))):
    #     for j, trans in enumerate(init_transitions + list(range(min_transitions, max_transitions, transition_interval))):
            
    #         seq_generator = RandomLetterSequenceGenerator(length=trans, repeat_pattern_len=state)
    #         eval_pipeline = EvaluationPipeline(model_name=model_name, sequence_generator=seq_generator)

    #         acc, correct_per_sample, sequences, labels = eval_pipeline.generate_and_classify_accuracy(num_samples=num_samples, seq_len=trans)

    #         for (correct,s,l) in zip(correct_per_sample, sequences, labels):
    #             if state not in correct_output:
    #                 correct_output[state] = {}
    #             if trans not in correct_output[state]:
    #                 correct_output[state][trans] = []
               
    #             if state not in incorrect_output:
    #                 incorrect_output[state] = {}
    #             if trans not in incorrect_output[state]:
    #                 incorrect_output[state][trans] = []
                

    #             if correct:
    #                 correct_output[state][trans].append([s, l])
    #             else:
    #                 incorrect_output[state][trans].append([s, l])
            
    #         accuracy_heatmap[i][j] = acc
    #         print(f"STATE {state} | TRANS {trans} | ACCURACY: {acc*100}%")
            
    # pdb.set_trace()
    # os.mkdir(os.path.join(BASE_PATH, 'random'))
    # #save all the outputs and heatmaps
    # with open(os.path.join(BASE_PATH, 'random', 'correct_random_letter.json'), 'w') as f:
    #     json.dump(correct_output, f, indent=4)
    # with open(os.path.join(BASE_PATH, 'random', 'incorrect_random_letter.json'), 'w') as f:
    #     json.dump(incorrect_output, f, indent=4)
    
    # # Optional: label the rows and columns
    # row_labels = init_states + [state for state in range(min_states, max_states, state_interval)]
    # col_labels = init_transitions + [trans for trans in range(min_transitions, max_transitions, transition_interval)]
    # df = pd.DataFrame(accuracy_heatmap, index=row_labels, columns=col_labels)

    # # Plot and save heatmap
    # plt.figure(figsize=(8, 8))
    # ax = sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", annot_kws={"size": 5, "rotation": 30})# Label colorbar
    # colorbar = ax.collections[0].colorbar
    # colorbar.set_label('Accuracy')
    # plt.title(f"Random Sequence | {model_name}")
    # plt.xlabel("Number of Transitions")
    # plt.ylabel("Number of States")
    # plt.savefig(os.path.join(BASE_PATH,'random', "accuracy_random_letter.png"), dpi=300)
    # plt.close()

    pdb.set_trace()

    #tracking correct and incorrect outputs
    correct_output = {}
    incorrect_output = {}
    #track 3D heatmaps of accuracy for DFA state action environment
    accuracy_heatmaps = np.zeros((len(init_states)+(max_states-min_states)//state_interval, len(init_transitions) + (max_transitions-min_transitions)//transition_interval, density_interval))

    for i, state in enumerate(init_states + list(range(min_states, max_states, state_interval))):
        for j, trans in enumerate(init_transitions + list(range(min_transitions, max_transitions, transition_interval))):
                
            max_density = max(1, factorial(state) / 2*factorial(state-2))
            interval = max((max_density-state+1)//density_interval, 1)
            for k, density in enumerate(range(state-1, max_density, interval)):
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
                pdb.set_trace()
                seq_generator = DFAStateActionSequenceGenerator(num_states=state, num_edges=density, num_unique_actions=state//2)
                eval_pipeline = EvaluationPipeline(model_name=model_name, sequence_generator=seq_generator)

                acc, correct_per_sample, sequences, labels = eval_pipeline.generate_and_classify_accuracy(num_samples=num_samples, seq_len=trans)

                for (correct,s,l) in zip(correct_per_sample, sequences, labels):
                    
                    if correct:
                        correct_output[state][trans][density].append([s, l])
                    else:
                        incorrect_output[state][trans][density].append([s, l])
                
                accuracy_heatmap[i][j][k] = acc
    #save all the outputs and heatmaps
    os.mkdir(os.path.join(BASE_PATH, 'dfa_state'))
    with open(os.path.join(BASE_PATH, 'dfa_state', 'correct_dfastateaction.json'), 'w') as f:
        json.dump(correct_output, f, indent=4)
    with open(os.path.join(BASE_PATH, 'dfa_state', 'incorrect_dfastateaction.json'), 'w') as f:
        json.dump(incorrect_output, f, indent=4)
    #plot and save 3D heatmap
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(accuracy_heatmap, facecolors='blue', edgecolor='k')
    # Add colorbar
    norm = Normalize(vmin=accuracy_heatmap.min(), vmax=accuracy_heatmap.max())
    colors = viridis(norm(accuracy_heatmap))
    mappable = ScalarMappable(norm=norm, cmap=viridis)
    mappable.set_array(accuracy_heatmap)
    cbar = plt.colorbar(mappable, ax=ax, pad=0.1)
    cbar.set_label("Accuracy")
    plt.savefig(os.path.join(BASE_PATH, 'dfa_state', "accuracy_dfaactionstate.png"), dpi=300)
    plt.close()
    

    #tracking correct and incorrect outputs
    correct_output = {}
    incorrect_output = {}
    #track 3D heatmaps of accuracy for DFA state state environment
    accuracy_heatmaps = np.zeros(( len(init_states)+ (max_states-min_states)//state_interval, len(init_transitions) + (max_transitions-min_transitions)//transition_interval, density_interval))

    for i, state in enumerate(init_states+list(range(min_states, max_states, state_interval))):
        for j, trans in enumerate(init_transitions + list(range(min_transitions, max_transitions, transition_interval))):
                
            max_density = max(factorial(state) / 2*factorial(state-2),1)
            interval = max((max_density-state+1)//density_interval, 1)

            for k, density in enumerate(range(state-1, max_density, interval)):

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
                
                seq_generator = DFAStateSequenceGenerator(num_states=state, num_edges=density)
                eval_pipeline = EvaluationPipeline(model_name=model_name, sequence_generator=seq_generator)

                acc, correct_per_sample, sequences, labels = eval_pipeline.generate_and_classify_accuracy(num_samples=num_samples, seq_len=trans)

                for (correct,s,l) in zip(correct_per_sample, sequences, labels):
                    if correct:
                        correct_output[state][trans][density].append([s, l])
                    else:
                        incorrect_output[state][trans][density].append([s, l])
                
                accuracy_heatmap[i][j][k] = acc
                pdb.set_trace()

    os.mkdir(os.path.join(BASE_PATH, 'dfa_state'))
    #save all the outputs and heatmaps
    with open(os.path.join(BASE_PATH, 'dfa_state','correct_dfastate.json'), 'w') as f:
        json.dump(correct_output, f, indent=4)
    with open(os.path.join(BASE_PATH, 'dfa_state', 'incorrect_dfastate.json'), 'w') as f:
        json.dump(incorrect_output, f, indent=4)
    
    #plot and save 3D heatmap
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(accuracy_heatmap, facecolors='blue', edgecolor='k')
    norm = Normalize(vmin=accuracy_heatmap.min(), vmax=accuracy_heatmap.max())
    colors = viridis(norm(accuracy_heatmap))
    mappable = ScalarMappable(norm=norm, cmap=viridis)
    mappable.set_array(accuracy_heatmap)
    cbar = plt.colorbar(mappable, ax=ax, pad=0.1)
    cbar.set_label('Accuracy')
    plt.title(f"Random Sequence | {model_name}")
    plt.xlabel("Number of Transitions")
    plt.ylabel("Number of States")
    plt.savefig(os.path.join(BASE_PATH, 'dfa_state', "accuracy_dfastate.png"), dpi=300)
    plt.close()

    
    #tracking correct and incorrect outputs
    correct_output = {}
    incorrect_output = {}
    #track 3D heatmaps of accuracy for DFA state action environment
    accuracy_heatmaps = np.zeros((len(init_states) + (max_states-min_states)//state_interval, len(init_transitions) + (max_transitions-min_transitions)//transition_interval), density_interval)

    for i, state in enumerate(init_states + list(range(min_states, max_states, state_interval))):
        for j, trans in enumerate(init_transitions + list(range(min_transitions, max_transitions, transition_interval))):
                
            max_density = factorial(state) / 2*factorial(state-2)
            for k, density in enumerate(range(state-1, max_density, (max_density-state+1)//density_interval)):

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
                
                seq_generator = DFAPDDLSequenceGenerator(num_states=state, num_edges=density, num_unique_actions=state//2)
                eval_pipeline = EvaluationPipeline(model_name=model_name, sequence_generator=seq_generator)

                acc, correct_per_sample, sequences, labels = eval_pipeline.generate_and_classify_accuracy(num_samples=num_samples, seq_len=trans)

                for (correct,s,l) in zip(correct_per_sample, sequences, labels):
                    
                    if correct:
                        correct_output[state][trans][density].append([s, l])
                    else:
                        incorrect_output[state][trans][density].append([s, l])
                
                accuracy_heatmap[i][j][k] = acc

        #save all the outputs and heatmaps
        os.mkdir(os.path.join(BASE_PATH, 'dfa_pddl'))
        with open(os.path.join(BASE_PATH, 'dfa_pddl', 'correct_dfapddl.json'), 'w') as f:
            json.dump(correct_output, f, indent=4)
        with open(os.path.join(BASE_PATH, 'dfa_pddl', 'incorrect_dfapddl.json'), 'w') as f:
            json.dump(incorrect_output, f, indent=4)
        #plot and save 3D heatmap
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.voxels(accuracy_heatmap, facecolors='blue', edgecolor='k')
         # Add colorbar
        norm = Normalize(vmin=accuracy_heatmap.min(), vmax=accuracy_heatmap.max())
        colors = viridis(norm(accuracy_heatmap))
        mappable = ScalarMappable(norm=norm, cmap=viridis)
        mappable.set_array(accuracy_heatmap)
        cbar = plt.colorbar(mappable, ax=ax, pad=0.1)
        cbar.set_label("Accuracy")
        plt.savefig(os.path.join(BASE_PATH, 'dfa_pddl', "accuracy_dfapddl.png"), dpi=300)
        plt.close()

                

            
            

            
            


            


# Example usage
# seq_generator = RandomLetterSequenceGenerator(...)
# pipeline = EvaluationPipeline('pythia-70m', seq_generator)
# accuracy = pipeline.generate_and_classify_accuracy(num_samples=100)
# print("Accuracy:", accuracy)
# token_probs = pipeline.get_token_probabilities("example sequence")
# print("Token Probabilities:", token_probs)
# activation_patterns = pipeline.get_activation_patterns("example sequence", layer_idx=3, head_idx=6)
# print("Activation Patterns:", activation_patterns)
# logits, log_probs = pipeline.get_logits_and_logprobs("example sequence")
# print("Logits:", logits)
# print("Log Probabilities:", log_probs) 