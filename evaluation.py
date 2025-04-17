import torch
import numpy as np
from transformer_lens import HookedTransformer, HookedTransformerConfig, ActivationCache
import transformer_lens.utils as utils
import circuitsvis as cv
from config import Config
import plotly.express as px

class EvaluationPipeline:
    def __init__(self, model_name:str):
        self.model = HookedTransformer.from_pretrained(
            model_name,
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=True,
            low_cpu_mem_usage=True
        )
        self.seq_generator = Config.sequence_generator

    def generate_and_classify_accuracy(self, num_samples:int=10, seq_len:int=10):
        # Generate sequences and classify accuracy
        correct = 0
        for _ in range(num_samples):
            sequence, label = self.seq_generator.generate(seq_len=seq_len)
            tokens = self.model.to_tokens(sequence, prepend_bos=True)
            logits, _ = self.model.run_with_cache(tokens)
            probs = torch.nn.functional.softmax(logits, dim=-1)[:, -1, :]
            pred_label = torch.argmax(probs, axis=-1)
            true_label = self.model.to_tokens(label, prepend_bos=False)
            true_label = torch.squeeze(true_label)
            if pred_label == true_label:
                correct += 1
        return correct / num_samples

    def get_token_probabilities(self, seq_len:int):
        # Return probabilities of likely next tokens
        sequence, label = self.seq_generator.generate(seq_len=seq_len)
        tokens = self.model.to_tokens(sequence, prepend_bos=True)
        logits, _ = self.model.run_with_cache(tokens)
        probs = torch.nn.functional.softmax(logits, dim=-1)[:, -1, :]
        return probs

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