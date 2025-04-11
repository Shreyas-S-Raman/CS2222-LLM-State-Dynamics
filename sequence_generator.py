import random
import string
import networkx as nx
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class SequenceGenerator(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def generate(self, length:int, repeat_pattern_len:int):
        pass
    
    @abstractmethod
    def visualize_dfa(self):
        #NOTE: use self.dfa to visualize the DFA
        pass


class RandomLetterSequenceGenerator(SequenceGenerator):
    def __init__(self, length:int=None, repeat_pattern_len:int=None, end_idx:int=None):
        super().__init__()
        self.length = length
        self.repeat_pattern_len = repeat_pattern_len
        self.end_idx = end_idx
        assert self.end_idx < self.repeat_pattern_len, "ERROR: end_idx must be less than repeat_pattern_len"

    def generate(self, length, repeat_pattern_len):
        repeat_pattern_len = repeat_pattern_len if self.repeat_pattern_len is None else self.repeat_pattern_len
        length = length if self.length is None else self.length
        base_pattern = random.choices(string.ascii_lowercase, k=repeat_pattern_len)
        repeats = (length) // repeat_pattern_len

        pattern = base_pattern * repeats
        if self.end_idx is not None and self.end_idx > 0:
            pattern += base_pattern[:self.end_idx]

        return pattern


class DFAStateActionSequenceGenerator(SequenceGenerator):
    def __init__(self, dfa=None, num_states:int=2, num_edges:int=1, num_unique_actions:int=1, max_sink_nodes:int=1, start_state:str=None):
        super().__init__()

        assert num_edges <= (self.factorial(num_states))/(2*self.factorial(num_states-2)), "ERROR: cannot have more edges than maximally possible between all nodes"
        assert max_sink_nodes < num_states, "ERROR: number of sink nodes cannot be more than all nodes"

        self.dfa = dfa if dfa is not None else self._create_dfa(num_states=num_states, num_edges=num_edges, num_unique_actions=num_unique_actions, max_sink_nodes=max_sink_nodes)

        assert start_state is None or start_state in self.dfa, "ERROR: start state is not in the DFA"

        self.start_state = start_state
    
    def generate(self, seq_len:int = None, start_state:str=None):

        curr_state = start_state if start_state is not None else self.start_state
        num_steps = 0

        #otuput sequence of states
        sequence = []
        
        while curr_state is not None and num_steps < seq_len:
            sequence.append(curr_state)

            if len(self.dfa[curr_state]) > 0:
                rand_action = random.choice(self.dfa[curr_state].keys()) 
                sequence.append(rand_action)
                curr_state = self.dfa[curr_state][rand_action]
            else:
                curr_state = None
            num_steps += 1
        
        return sequence

    def visualize_dfa(self):
        G = nx.DiGraph()

        for state, transitions in self.dfa.items():
            for action, next_state in transitions.items():
                G.add_edge(state, next_state, label=action)

        pos = nx.spring_layout(G)  # You can use other layouts like circular or planar

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1500, edgecolors='black')
        nx.draw_networkx_labels(G, pos)

        # Draw edges
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edges(G, pos, arrows=True)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.axis('off')
        plt.show()


    def _create_dfa(self, num_states:int, num_edges:int, num_unique_actions:int, max_sink_nodes:int):

        for i in range(0, num_states//len(string.ascii_lowercase)+1):
            all_states = [s + f'_{i}' for s in string.ascii_lowercase]
        for i in range(0, num_unique_actions//len(string.ascii_uppercase)+1):
            all_actions = [a + f'_{i}' for a in string.ascii_uppercase]
        
        all_states = set(all_states)
        all_actions = set(all_actions)

        #populate the set of actions and states used to make the DFA
        self.states = random.sample(all_states, num_states)
        self.actions = set(random.sample(all_actions, num_unique_actions))

        #assign relevant actions to each edge
        distributed_edges = self._distribute_edges(num_edges=num_edges, num_states=num_states, max_sink_nodes=max_sink_nodes)
        self.dfa = {} #{s1: {action: s2}}

        for i, num_out_edges in enumerate(distributed_edges):

            
            sampled_actions = random.sample(self.actions, num_out_edges)
            sampled_states = random.sample(self.states, num_out_edges)
            
            self.dfa[self.states[i]] = {a:s for (a,s) in zip(sampled_actions, sampled_states)}
    
    def _distribute_edges(self, num_edges: int, num_states: int, max_sink_nodes: int):
        if num_states <= 0:
            raise ValueError("Number of states must be positive.")
        if num_edges < 0:
            raise ValueError("Number of edges must be non-negative.")
        if max_sink_nodes >= num_states:
            raise ValueError("max_sink_nodes must be less than num_states.")

        # Step 1: Randomly select number of sink nodes (up to max)
        num_sinks = random.randint(1, max_sink_nodes)
        sink_indices = set(random.sample(range(num_states), num_sinks))

        # Step 2: Allocate edges to the rest
        non_sink_states = num_states - num_sinks
        if num_edges == 0:
            return [0] * num_states

        # Reuse stars and bars on non-sink states
        cut_points = sorted(random.sample(range(num_edges + non_sink_states - 1), non_sink_states - 1))
        cut_points = [-1] + cut_points + [num_edges + non_sink_states - 1]
        non_sink_edges = [cut_points[i+1] - cut_points[i] - 1 for i in range(non_sink_states)]

        # Step 3: Combine into full list
        result = []
        non_sink_ptr = 0
        for i in range(num_states):
            if i in sink_indices:
                result.append(0)
            else:
                result.append(non_sink_edges[non_sink_ptr])
                non_sink_ptr += 1

        return result

    def factorial(self, n:int):
        if n==1 or n==0:
            return 1
        return n * self.factorial(n-1)
                             
class DFAStateSequenceGenerator(SequenceGenerator):

    def __init__(self, dfa=None, num_states:int=2, num_edges:int=1, max_sink_nodes:int=1, start_state:str=None):
        super().__init__()

        assert num_edges <= (self.factorial(num_states))/(2*self.factorial(num_states-2)), "ERROR: cannot have more edges than maximally possible between all nodes"
        assert max_sink_nodes < num_states, "ERROR: number of sink nodes cannot be more than all nodes"

        self.dfa = dfa if dfa is not None else self._create_dfa(num_states=num_states, num_edges=num_edges, max_sink_nodes=max_sink_nodes)

        assert start_state is None or start_state in self.dfa, "ERROR: start state is not in the DFA"
        self.start_state = start_state

    def generate(self, seq_len:int = None, start_state:str=None):
        curr_state = start_state if start_state is not None else self.start_state
        num_steps = 0

        #otuput sequence of states
        sequence = []
        
        while curr_state is not None and num_steps < seq_len:
            sequence.append(curr_state)
            if len(self.dfa[curr_state]) > 0:
                rand_action = random.choice(self.dfa[curr_state].keys()) 
                sequence.append(rand_action)
                curr_state = self.dfa[curr_state][rand_action]
            else:
                curr_state = None
            num_steps += 1
        
        return sequence

    def _create_dfa(self, num_states:int, num_edges:int, max_sink_nodes:int):

        for i in range(0, num_states//len(string.ascii_lowercase)+1):
            all_states = [s + f'_{i}' for s in string.ascii_lowercase]
       
        all_states = set(all_states)
        all_actions = set(all_actions)

        #populate the set of states used to make the DFA
        self.states = random.sample(all_states, num_states)

        #assign relevant actions to each edge
        distributed_edges = self._distribute_edges(num_edges=num_edges, num_states=num_states, max_sink_nodes=max_sink_nodes)
        self.dfa = {} #{s1: set{s2}}


        for i, num_out_edges in enumerate(distributed_edges):

            sampled_states = random.sample(self.states, num_out_edges)

            self.dfa[self.states[i]] = set([s for s in sampled_states])
    
    def _distribute_edges(self, num_edges: int, num_states: int, max_sink_nodes: int):
        if num_states <= 0:
            raise ValueError("Number of states must be positive.")
        if num_edges < 0:
            raise ValueError("Number of edges must be non-negative.")
        if max_sink_nodes >= num_states:
            raise ValueError("max_sink_nodes must be less than num_states.")

        # Step 1: Randomly select number of sink nodes (up to max)
        num_sinks = random.randint(1, max_sink_nodes)
        sink_indices = set(random.sample(range(num_states), num_sinks))

        # Step 2: Allocate edges to the rest
        non_sink_states = num_states - num_sinks
        if num_edges == 0:
            return [0] * num_states

        # Reuse stars and bars on non-sink states
        cut_points = sorted(random.sample(range(num_edges + non_sink_states - 1), non_sink_states - 1))
        cut_points = [-1] + cut_points + [num_edges + non_sink_states - 1]
        non_sink_edges = [cut_points[i+1] - cut_points[i] - 1 for i in range(non_sink_states)]

        # Step 3: Combine into full list
        result = []
        non_sink_ptr = 0
        for i in range(num_states):
            if i in sink_indices:
                result.append(0)
            else:
                result.append(non_sink_edges[non_sink_ptr])
                non_sink_ptr += 1

        return result

    def factorial(self, n:int):
        if n==1 or n==0:
            return 1
        return n * self.factorial(n-1)

    def visualize_dfa(self):
        G = nx.DiGraph()

        for state, next_states in self.dfa.items():
            for next_state in next_states:
                G.add_edge(state, next_state)

        pos = nx.spring_layout(G)  # Layout can be adjusted if needed

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1500, edgecolors='black')
        nx.draw_networkx_labels(G, pos)

        # Draw edges
        nx.draw_networkx_edges(G, pos, arrows=True)

        plt.axis('off')
        plt.show()