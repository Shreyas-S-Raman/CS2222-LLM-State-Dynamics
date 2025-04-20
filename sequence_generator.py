import random
import string
import networkx as nx
import warnings
warnings.filterwarnings("ignore", message="networkx backend defined more than once")
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import pdb

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
    def __init__(self, length:int=None, repeat_pattern_len:int=None, end_idx:int=None, randomize:bool=False):
        super().__init__()
        self.randomize = randomize
        self.length = length
        self.repeat_pattern_len = repeat_pattern_len
        self.end_idx = end_idx
        assert self.end_idx < self.repeat_pattern_len if end_idx else True, "ERROR: end_idx must be less than repeat_pattern_len"

    def generate(self, seq_len:int=None, repeat_pattern_len:int=None):
        #NOTE: edges = seq_len / length | states = repeat_pattern_len
        repeat_pattern_len = repeat_pattern_len if self.repeat_pattern_len is None else self.repeat_pattern_len
        length = seq_len if self.length is None else self.length
        base_pattern = random.choices(string.ascii_lowercase, k=repeat_pattern_len)
        repeats = max((length) // repeat_pattern_len,1)

        pattern = base_pattern * repeats

        if self.end_idx is not None and self.end_idx > 0:
            pattern += base_pattern[:self.end_idx]
        elif self.randomize:
            end_idx = random.choice(range(0,repeat_pattern_len))
            pattern += base_pattern[:end_idx]
        elif repeat_pattern_len > length:
            end_idx = length
            pattern = pattern[:end_idx]

        # Final sequence of desired length
        sequence = pattern[:length]

        # Find index of the last character in base_pattern
        last_char = sequence[-1]
        
        try:
            last_idx = base_pattern.index(last_char)
        except ValueError:
            raise ValueError("Last character not in base_pattern")

        # Wrap-around next token
        label = base_pattern[(last_idx + 1) % len(base_pattern)]

        return ' '.join(sequence), ' '+label
    
    def visualize_dfa(self):
        pass

class DFAStateActionSequenceGenerator(SequenceGenerator):
    def __init__(self, dfa=None, num_states:int=2, num_edges:int=1, num_unique_actions:int=1, max_sink_nodes:int=1, start_state:str=None, reduce_states: bool = False):
        super().__init__()
        #NOTE: some 'states' (2 lower case letter combos) registered as 2 tokens making next state prediction not valuable
        self.skip_states = {'gv', 'vz', 'gx', 'lm', 'cq', 'zj', 'sz', 'kv', 'zv', 'qb', 'xz', 'rj', 'dz', 'pq', 'hf', 'wv', 'rk', 'vf', 'wj', 'xm', 'qk', 'zk', 'hz', 'gj', 'vw', 'tq', 'zf', 'hc', 'jd', 'sj', 'zw', 'lg', 'xg', 'zc', 'ej', 'hk', 'qj', 'tb', 'jx', 'wz', 'lw', 'mq', 'lj', 'jr', 'kq', 'vn', 'hv', 'gf', 'xr', 'qe', 'zs', 'vk', 'qm', 'jf', 'rz', 'qo', 'jk', 'xv', 'jz', 'yw', 'fz', 'bv', 'nf', 'kp', 'xu', 'kz', 'yj', 'mf', 'hb', 'gq', 'zr', 'jm', 'gk', 'jh', 'wc', 'xk', 'pv', 'fj', 'jb', 'mw', 'mh', 'nq', 'jv', 'xh', 'kc', 'bw', 'hg', 'dg', 'xq', 'tj', 'kx', 'qr', 'lk', 'fh', 'uq', 'pf', 'jg', 'kd', 'gw', 'qf', 'jq', 'yf', 'cg', 'zp', 'qz', 'xo', 'mj', 'xl', 'cx', 'vb', 'zm', 'sx', 'fq', 'qp', 'wg', 'yb', 'hx', 'dw', 'bq', 'jw', 'qh', 'uw', 'lz', 'nh', 'bk', 'qv', 'bz', 'qg', 'lq', 'zt', 'zq', 'hq', 'lx', 'pk', 'xj', 'cj', 'rv', 'wf', 'jy', 'nj', 'dv', 'eo', 'yh', 'pz', 'kf', 'qx', 'fv', 'zd', 'oq', 'zl', 'vh', 'rp', 'yc', 'qd', 'yv', 'mv', 'yq', 'cw', 'qn', 'vq', 'wq', 'xw', 'hj', 'mz', 'pj', 'lh', 'vx', 'bx', 'fk', 'zx', 'qy', 'qw', 'qc', 'rq', 'nk', 'jt', 'xn', 'jn', 'pw', 'zg', 'nx', 'vj'}
        #NOTE: some 'actions' (2 upper case letter combos) registerd as 2 tokens making next state predction not valuable
        self.skip_actions = {'YC', 'LQ', 'YM', 'HG', 'VU', 'ZM', 'HW', 'NQ', 'ZH', 'FZ', 'LJ', 'FQ', 'CY', 'YU', 'KU', 'WU', 'BK', 'LW', 'YQ', 'YD', 'RK', 'DY', 'QM', 'QH', 'QZ', 'RV', 'QC', 'JH', 'ZL', 'SV', 'AJ', 'WL', 'VF', 'WV', 'NJ', 'ZD', 'XB', 'JX', 'QJ', 'KD', 'ZY', 'QX', 'XN', 'JN', 'VX', 'EJ', 'GK', 'IH', 'QN', 'CZ', 'KG', 'EZ', 'QE', 'LK', 'GZ', 'YB', 'XR', 'KL', 'JU', 'XO', 'XZ', 'QP', 'ZQ', 'QT', 'YX', 'TZ', 'VK', 'ZJ', 'XH', 'YT', 'CX', 'RU', 'BJ', 'WX', 'QR', 'XD', 'ZF', 'UH', 'VV', 'ZW', 'TJ', 'NU', 'RX', 'XA', 'GX', 'SJ', 'LN', 'MV', 'HZ', 'PZ', 'MJ', 'AO', 'XC', 'JE', 'WI', 'ZB', 'YG', 'QK', 'KV', 'ZP', 'ZG', 'LH', 'PQ', 'VY', 'QB', 'JW', 'JF', 'UJ', 'KJ', 'GJ', 'ZV', 'XU', 'YW', 'ZT', 'KQ', 'ZI', 'EI', 'XE', 'DV', 'JK', 'AQ', 'WZ', 'EQ', 'XF', 'OI', 'BX', 'MZ', 'JQ', 'XK', 'DG', 'TU', 'QI', 'JV', 'KM', 'KP', 'IW', 'QA', 'RQ', 'CQ', 'LZ', 'QW', 'HV', 'UY', 'YZ', 'DZ', 'NZ', 'OQ', 'NX', 'YF', 'HX', 'LX', 'JZ', 'QS', 'YV', 'ZC', 'QD', 'ZU', 'QV', 'RJ', 'VN', 'WQ', 'DQ', 'EK', 'ZO', 'VH', 'VJ', 'UZ', 'EO', 'OZ', 'QY', 'BV', 'SX', 'YI', 'VW', 'PV', 'PJ', 'JT', 'KW', 'YO', 'OJ', 'JY', 'DU', 'WJ', 'KX', 'XG', 'WO', 'PX', 'TQ', 'UW', 'VZ', 'JL', 'SZ', 'XI', 'BZ', 'RZ', 'XV', 'HJ', 'XS', 'WY', 'YJ', 'CK', 'KF', 'JI', 'UO', 'BQ', 'ZS', 'TK', 'EY', 'GQ', 'BH', 'YH', 'QO', 'YK', 'LF', 'QQ', 'CJ', 'SQ', 'KZ', 'HN', 'ZK', 'FJ', 'WK', 'KH', 'QG', 'QF', 'HU', 'PY', 'WG', 'UF', 'XJ', 'VQ', 'JG', 'UQ', 'XW', 'IY', 'ZN', 'VG', 'FV', 'PW', 'ZR', 'XQ'}

        assert num_edges <= num_states*(num_states-1)//2 + 1, "ERROR: cannot have more edges than maximally possible between all nodes"
        assert num_edges >= num_states, "ERROR: there must be an edge at least between every state to ensure no state isolated"
        assert max_sink_nodes <= num_states, "ERROR: number of sink nodes cannot be more than all nodes"

        self.num_states = num_states
        self.num_edges = num_edges
        self.num_unique_actions = num_unique_actions
        self.max_sink_nodes = max_sink_nodes
        self.reduce_states = reduce_states
        self.dfa = dfa if dfa is not None else self._create_dfa(num_states=num_states, num_edges=num_edges, num_unique_actions=num_unique_actions, max_sink_nodes=max_sink_nodes)
        assert start_state is None or start_state in self.dfa, "ERROR: start state is not in the DFA"
        self.start_state = start_state
    
    def generate_with_curr_dfa(self, seq_len:int = None, start_state:str=None):
        if start_state is not None:
            curr_state = start_state
        elif self.start_state is not None:
            curr_state = self.start_state
        else:
            curr_state = random.choice(list(self.dfa.keys()))

        num_steps = 0
        sequence = []
        while curr_state is not None and num_steps < seq_len:
            sequence.append(curr_state)
            num_steps += 1
            
            if self.dfa[curr_state]:
                rand_action = random.choice(list(self.dfa[curr_state].keys()))
                sequence.append(rand_action)
                curr_state = self.dfa[curr_state][rand_action]
            else:
                break
           
        label = curr_state

        return ' '.join(sequence) + ' ' if not self.reduce_states else ' '.join(sequence), label if not self.reduce_states else ' '+label

    def generate(self, seq_len:int = None, start_state:str=None):
        self.dfa = self._create_dfa(num_states=self.num_states, num_edges=self.num_edges, num_unique_actions=self.num_unique_actions, max_sink_nodes=self.max_sink_nodes)
        if start_state is not None:
            curr_state = start_state
        elif self.start_state is not None:
            curr_state = self.start_state
        else:
            curr_state = random.choice(list(self.dfa.keys()))

        num_steps = 0
        sequence = []
        while curr_state is not None and num_steps < seq_len:
            sequence.append(curr_state)
            num_steps += 1
            
            if self.dfa[curr_state]:
                rand_action = random.choice(list(self.dfa[curr_state].keys()))
                sequence.append(rand_action)
                curr_state = self.dfa[curr_state][rand_action]
            else:
                break
        
        label = curr_state

        return ' '.join(sequence) + ' ' if not self.reduce_states else ' '.join(sequence), label if not self.reduce_states else ' '+label

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
        if self.reduce_states:
            all_states = set()
            for s in string.ascii_lowercase:
                all_states.add(s)
            all_actions = set()
            for a in string.ascii_uppercase:
                all_actions.add(a)
        else:
            all_states = set()
            for c1 in string.ascii_lowercase:
                for c2 in string.ascii_lowercase:
                    all_states.add(f'{c1}{c2}')
            all_actions = set()
            for c1 in string.ascii_uppercase:
                for c2 in string.ascii_uppercase:
                    all_actions.add(f'{c1}{c2}')
            all_states = all_states - self.skip_states
            all_actions = all_actions - self.skip_actions

        # Populate the set of actions and states used to make the DFA
        self.states = random.sample(all_states, num_states)
        self.actions = set(random.sample(all_actions, num_unique_actions))

        # Assign relevant actions to each edge
        distributed_edges = self._distribute_edges(num_edges=num_edges, num_states=num_states, max_sink_nodes=max_sink_nodes)
        dfa = {}  # {s1: {action: s2}}

        for i, num_out_edges in enumerate(distributed_edges):
            sampled_actions = random.sample(self.actions, num_out_edges)
            sampled_states = random.choices(self.states, k=num_out_edges)
            
            # Ensure each state has a complete set of actions
            dfa[self.states[i]] = {a: s for (a, s) in zip(sampled_actions, sampled_states)}

            # Ensure each sink node has a self-looping edge
            if num_out_edges == 0:
                if self.states[i] not in dfa:
                    dfa[self.states[i]] = {}
                if len(self.actions) > 0:
                    self_loop_action = random.choice(list(self.actions))
                    dfa[self.states[i]][self_loop_action] = self.states[i]

        return dfa
    
    def _distribute_edges(self, num_edges: int, num_states: int, max_sink_nodes: int):
        if num_states <= 0:
            raise ValueError("Number of states must be positive.")
        if num_edges < 0:
            raise ValueError("Number of edges must be non-negative.")
        if max_sink_nodes > num_states:
            raise ValueError("max_sink_nodes must be less than num_states.")

        # Step 1: Randomly select number of sink nodes (up to max)
        num_sinks = random.randint(1, max_sink_nodes)
        sink_indices = set(random.sample(range(num_states), num_sinks))

        # Step 2: Allocate edges to the rest
        non_sink_states = num_states - num_sinks
        if num_edges == 0:
            return [0] * num_states

        # Ensure each state has at least one outgoing edge
        min_outgoing_edges = 1
        remaining_edges = num_edges - num_states  # Subtract the minimum required edges

        # Distribute remaining edges with a maximum of num_states per node
        non_sink_edges = [min_outgoing_edges] * num_states
        for i in range(remaining_edges):
            non_sink_edges[i % num_states] += 1

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

class DFAStateSequenceGenerator(SequenceGenerator):
    def __init__(self, dfa=None, num_states:int=2, num_edges:int=1, max_sink_nodes:int=1, start_state:str=None, reduce_states: bool = False):
        super().__init__()
        #NOTE: some 'states' (2 lower case letter combos) registered as 2 tokens making next state prediction not valuable
        self.skip_states = {'gv', 'vz', 'gx', 'lm', 'cq', 'zj', 'sz', 'kv', 'zv', 'qb', 'xz', 'rj', 'dz', 'pq', 'hf', 'wv', 'rk', 'vf', 'wj', 'xm', 'qk', 'zk', 'hz', 'gj', 'vw', 'tq', 'zf', 'hc', 'jd', 'sj', 'zw', 'lg', 'xg', 'zc', 'ej', 'hk', 'qj', 'tb', 'jx', 'wz', 'lw', 'mq', 'lj', 'jr', 'kq', 'vn', 'hv', 'gf', 'xr', 'qe', 'zs', 'vk', 'qm', 'jf', 'rz', 'qo', 'jk', 'xv', 'jz', 'yw', 'fz', 'bv', 'nf', 'kp', 'xu', 'kz', 'yj', 'mf', 'hb', 'gq', 'zr', 'jm', 'gk', 'jh', 'wc', 'xk', 'pv', 'fj', 'jb', 'mw', 'mh', 'nq', 'jv', 'xh', 'kc', 'bw', 'hg', 'dg', 'xq', 'tj', 'kx', 'qr', 'lk', 'fh', 'uq', 'pf', 'jg', 'kd', 'gw', 'qf', 'jq', 'yf', 'cg', 'zp', 'qz', 'xo', 'mj', 'xl', 'cx', 'vb', 'zm', 'sx', 'fq', 'qp', 'wg', 'yb', 'hx', 'dw', 'bq', 'jw', 'qh', 'uw', 'lz', 'nh', 'bk', 'qv', 'bz', 'qg', 'lq', 'zt', 'zq', 'hq', 'lx', 'pk', 'xj', 'cj', 'rv', 'wf', 'jy', 'nj', 'dv', 'eo', 'yh', 'pz', 'kf', 'qx', 'fv', 'zd', 'oq', 'zl', 'vh', 'rp', 'yc', 'qd', 'yv', 'mv', 'yq', 'cw', 'qn', 'vq', 'wq', 'xw', 'hj', 'mz', 'pj', 'lh', 'vx', 'bx', 'fk', 'zx', 'qy', 'qw', 'qc', 'rq', 'nk', 'jt', 'xn', 'jn', 'pw', 'zg', 'nx', 'vj'}
        assert num_edges <= num_states*(num_states-1)//2 + 1, "ERROR: cannot have more edges than maximally possible between all nodes"
        assert num_edges >= num_states, "ERROR: there must be an edge at least between every state to ensure no state isolated"
        assert max_sink_nodes <= num_states, "ERROR: number of sink nodes cannot be more than all nodes"

        self.num_states = num_states
        self.num_edges = num_edges
        self.max_sink_nodes = max_sink_nodes
        self.reduce_states = reduce_states
        self.dfa = dfa if dfa is not None else self._create_dfa(num_states=num_states, num_edges=num_edges, max_sink_nodes=max_sink_nodes)
        assert start_state is None or start_state in self.dfa, "ERROR: start state is not in the DFA"
        self.start_state = start_state
    
    def generate_with_curr_dfa(self, seq_len:int = None, start_state:str=None):
        if start_state is not None:
            curr_state = start_state
        elif self.start_state is not None:
            curr_state = self.start_state
        else:
            curr_state = random.choice(list(self.dfa.keys()))

        num_steps = 0
        sequence = []
        while curr_state is not None and num_steps < seq_len:
            sequence.append(curr_state)
            num_steps += 1
            
            if self.dfa[curr_state]:
                curr_state = random.choice(self.dfa[curr_state])
            else:
                break
           
        label = curr_state

        return ' '.join(sequence) + ' ' if not self.reduce_states else ' '.join(sequence), label if not self.reduce_states else ' '+label

    def generate(self, seq_len:int = None, start_state:str=None):
        self.dfa = self._create_dfa(num_states=self.num_states, num_edges=self.num_edges, max_sink_nodes=self.max_sink_nodes)
        if start_state is not None:
            curr_state = start_state
        elif self.start_state is not None:
            curr_state = self.start_state
        else:
            curr_state = random.choice(list(self.dfa.keys()))

        num_steps = 0
        sequence = []
        while curr_state is not None and num_steps < seq_len:
            sequence.append(curr_state)
            num_steps += 1
            if self.dfa[curr_state]:
                curr_state = random.sample(self.dfa[curr_state],1)[0]
            else:
                break
           
        label = curr_state

        return ' '.join(sequence) + ' ' if not self.reduce_states else ' '.join(sequence), label if not self.reduce_states else ' '+label

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


    def _create_dfa(self, num_states:int, num_edges:int, max_sink_nodes:int):
        if self.reduce_states:
            all_states = set()
            for s in string.ascii_lowercase:
                all_states.add(s)
            
        else:
            all_states = set()
            for c1 in string.ascii_lowercase:
                for c2 in string.ascii_lowercase:
                    all_states.add(f'{c1}{c2}')
            all_states = all_states - self.skip_states

        # Populate the set of actions and states used to make the DFA
        self.states = random.sample(all_states, num_states)

        # Assign relevant actions to each edge
        distributed_edges = self._distribute_edges(num_edges=num_edges, num_states=num_states, max_sink_nodes=max_sink_nodes)
        dfa = {}  # {s1: {action: s2}}

        for i, num_out_edges in enumerate(distributed_edges):
            sampled_states = random.choices(self.states, k=num_out_edges)
            
            dfa[self.states[i]] = set([s for s in sampled_states])
            # Ensure each sink node has a self-looping edge
            if num_out_edges == 0:
                dfa[self.states[i]] = set([self.states[i]])

        return dfa
    
    def _distribute_edges(self, num_edges: int, num_states: int, max_sink_nodes: int):
        if num_states <= 0:
            raise ValueError("Number of states must be positive.")
        if num_edges < 0:
            raise ValueError("Number of edges must be non-negative.")
        if max_sink_nodes > num_states:
            raise ValueError("max_sink_nodes must be less than num_states.")

        # Step 1: Randomly select number of sink nodes (up to max)
        num_sinks = random.randint(1, max_sink_nodes)
        sink_indices = set(random.sample(range(num_states), num_sinks))

        # Step 2: Allocate edges to the rest
        non_sink_states = num_states - num_sinks
        if num_edges == 0:
            return [0] * num_states

        # Ensure each state has at least one outgoing edge
        min_outgoing_edges = 1
        remaining_edges = num_edges - num_states  # Subtract the minimum required edges

        # Distribute remaining edges with a maximum of num_states per node
        non_sink_edges = [min_outgoing_edges] * num_states
        for i in range(remaining_edges):
            non_sink_edges[i % num_states] += 1

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















#NOTE: unused class
# class DFAPDDLSequenceGenerator(SequenceGenerator):
#     def __init__(self, dfa=None, num_states: int = 2, num_edges: int = 1, num_unique_actions: int = 1,
#                  max_sink_nodes: int = 1, start_state: str = None):
#         super().__init__()
#         assert num_edges <= max(1, ((num_states*(num_states-1))//2)), "Too many edges"
#         assert num_edges >= num_states - 1, "ERROR: there must be an edge at least between every state to ensure no state isolated"
#         assert max_sink_nodes <= num_states, "ERROR: number of sink nodes cannot be more than all nodes"
        
#         self.state_vars = {f"flag_{i}": random.choice([0, 1]) for i in range(3)}  # binary state variables
#         self.action_definitions = {}  # action -> {'pre': ..., 'eff': ...}

#         self.dfa = dfa if dfa is not None else self._create_dfa(
#             num_states=num_states,
#             num_edges=num_edges,
#             num_unique_actions=num_unique_actions,
#             max_sink_nodes=max_sink_nodes
#         )

#         assert start_state is None or start_state in self.dfa, "Invalid start state"
#         self.start_state = start_state if start_state else random.choice(list(self.dfa.keys()))

#     def generate(self, seq_len: int = None, start_state: str = None):
#         curr_state = start_state if start_state is not None else self.start_state
#         num_steps = 0
#         sequence = []

#         while curr_state is not None and num_steps < seq_len:
#             transitions = self.dfa[curr_state]
#             valid_actions = [a for a in transitions if self._check_preconditions(a)]
#             if not valid_actions:
#                 break

#             # Select the action but do NOT apply it yet
#             action = random.choice(valid_actions)

#             # Log state_vars and action before effect
#             sequence.append(str(self.state_vars.copy()))
#             sequence.append(action)

#             # Compute label as the next state
#             label = transitions[action]

#             # Apply effects and transition
#             self._apply_effects(action)
#             curr_state = label
#             num_steps += 1

#         return ' '.join(sequence), ' ' +str(label)

#     def _check_preconditions(self, action):
#         return all(self.state_vars.get(k, 0) == v for k, v in self.action_definitions[action]['pre'].items())

#     def _apply_effects(self, action):
#         for k, v in self.action_definitions[action]['eff'].items():
#             self.state_vars[k] = v

#     def _create_dfa(self, num_states: int, num_edges: int, num_unique_actions: int, max_sink_nodes: int):
#         states = [f"s{i}" for i in range(num_states)]
#         actions = [f"a{i}" for i in range(num_unique_actions)]
#         self.states = states
#         self.actions = set(actions)

#         edge_counts = self._distribute_edges(num_edges, num_states, max_sink_nodes)
#         dfa = {}

#         for i, out_deg in enumerate(edge_counts):
#             source = states[i]
#             dfa[source] = {}
#             if out_deg == 0:
#                 continue

#             sampled_actions = random.sample(actions, out_deg)
#             target_states = random.sample(states, out_deg)

#             for action, target in zip(sampled_actions, target_states):
#                 dfa[source][action] = target

#                 # Define action preconditions/effects randomly
#                 pre = {f"flag_{j}": random.choice([0, 1]) for j in range(2)}
#                 eff = {f"flag_{j}": random.choice([0, 1]) for j in range(2)}
#                 self.action_definitions[action] = {'pre': pre, 'eff': eff}

#             # Ensure each sink node has a self-looping edge
#             if out_deg == 0:
#                 if source not in dfa:
#                     dfa[source] = {}
#                 if len(self.actions) > 0:
#                     self_loop_action = random.choice(list(self.actions))
#                     dfa[source][self_loop_action] = source

#         return dfa

#     def _distribute_edges(self, num_edges: int, num_states: int, max_sink_nodes: int):
#         if num_states <= 0 or num_edges < 0:
#             raise ValueError("States and edges must be non-negative")

#         num_sinks = random.randint(1, max_sink_nodes)
#         sink_indices = set(random.sample(range(num_states), num_sinks))
#         non_sink = num_states - num_sinks

#         if num_edges == 0:
#             return [0] * num_states

#         # Stars and bars for non-sink states
#         cut_points = sorted(random.sample(range(num_edges + non_sink - 1), non_sink - 1))
#         cut_points = [-1] + cut_points + [num_edges + non_sink - 1]
#         edge_alloc = [cut_points[i + 1] - cut_points[i] - 1 for i in range(non_sink)]

#         result = []
#         ptr = 0
#         for i in range(num_states):
#             result.append(0 if i in sink_indices else edge_alloc[ptr])
#             if i not in sink_indices:
#                 ptr += 1

#         return result

#     def visualize_dfa(self):
#         G = nx.DiGraph()
#         for state, transitions in self.dfa.items():
#             for action, next_state in transitions.items():
#                 G.add_edge(state, next_state, label=action)

#         pos = nx.spring_layout(G)
#         nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1500, edgecolors='black')
#         nx.draw_networkx_labels(G, pos)
#         nx.draw_networkx_edges(G, pos, arrows=True)
#         edge_labels = nx.get_edge_attributes(G, 'label')
#         nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
#         plt.axis('off')
#         plt.show()
   