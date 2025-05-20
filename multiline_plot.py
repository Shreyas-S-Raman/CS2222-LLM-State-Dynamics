import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from config import Config
import pdb
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, viridis

def scan_accuracy_curves():

    init_states = Config.init_states
    max_states = Config.max_states
    min_states = Config.min_states
    state_interval = Config.state_interval
    init_transitions = Config.init_transitions
    max_transitions = Config.max_transitions
    min_transitions = Config.min_transitions
    transition_interval = Config.transition_interval
    Config.model_name = 'pythia-1b'

    state_range = init_states+list(range(min_states, max_states, state_interval))
    trans_range = init_transitions + list(range(min_transitions, max_transitions, transition_interval))
    pdb.set_trace()
    accuracy_array = np.load('./pythia-1b_reduced/dfa_stateaction/accuracy_dfastateaction.npz')['array']

    # Get the shape of the heatmap
    num_states, num_transitions, density_intervals = accuracy_array.shape

    # Create a figure for each density interval
    for i, density in  enumerate([0.0, 0.5, 1.0]):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Extract the 2D slice for the current density interval
        accuracy_slice = accuracy_array[:, :, i].T
        
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
        plt.savefig(os.path.join(Config.BASE_PATH, 'dfa_stateaction', f"accuracy_dfastateaction_density_{density}.png"), dpi=300)
        plt.close()
    pdb.set_trace()
    accuracy_array = accuracy_array[:,:,1]*100

    # Plot random baseline
    baseline = [100 / n for n in state_range]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(state_range, baseline, '--', color='gray', label='Random Baseline', alpha=0.7)
    
    # Color map for transition curves
    colors = plt.cm.viridis(np.linspace(0, 1, len(trans_range)))
    
    # Plot accuracy curves for each transition count
    for i, num_transitions in enumerate(trans_range):
        if i%2 != 0:
            continue
        ax.plot(
            state_range, 
            accuracy_array[:,i], 
            'o-', 
            color=colors[i], 
            label=f'{num_transitions} Transition{"s" if num_transitions > 1 else ""}'
        )
    
    ax.set_xticks(state_range)
    ax.set_xlabel('# states', fontsize=20)
    ax.set_ylabel('acc (%)', fontsize=20)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    
    ax.legend(loc='best', fontsize=13)
    
    plt.tight_layout()
    plt.savefig('./pythia-1b_reduced/dfa_statestate/accuracy_lineplot.png')
    return fig

if __name__ == '__main__':
    scan_accuracy_curves()