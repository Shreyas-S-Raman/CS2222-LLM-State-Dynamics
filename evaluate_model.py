import argparse
from config import Config
from your_module import (
    evaluate_random_sequence,
    evaluate_dfa_stateaction_sequence,
    evaluate_dfa_statestate_sequence
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, help='Model name override')
    parser.add_argument('-r', '--reduce_states', action='store_true', help='Enable state reduction')
    args = parser.parse_args()

    # Override Config if CLI args provided
    if args.model_name:
        Config.model_name = args.model_name
    if args.reduce_states:
        Config.reduce_states = True
    
    path = f'./{model_name}' if not reduce_states else f'./{model_name}_reduced'
    Config.BASE_PATH = os.path.relpath(path)
    
    if not os.path.exists(Config.BASE_PATH):
        os.mkdir(Config.BASE_PATH)

    evaluate_random_sequence(
        model_name=Config.model_name, 
        num_samples=Config.num_samples, 
        init_states=Config.init_states, 
        init_transitions=Config.init_transitions, 
        max_states=Config.max_states, 
        min_states=Config.min_states, 
        state_interval=Config.state_interval, 
        max_transitions=Config.max_transitions, 
        min_transitions=Config.min_transitions, 
        transition_interval=Config.transition_interval
    )

    evaluate_dfa_stateaction_sequence(
        model_name=Config.model_name, 
        num_samples=Config.num_samples, 
        init_states=Config.init_states, 
        init_transitions=Config.init_transitions, 
        max_states=Config.max_states, 
        min_states=Config.min_states, 
        state_interval=Config.state_interval, 
        max_transitions=Config.max_transitions, 
        min_transitions=Config.min_transitions, 
        transition_interval=Config.transition_interval,
        density_interval=Config.density_interval,
        reduce_states=Config.reduce_states
    )

    evaluate_dfa_statestate_sequence(
        model_name=Config.model_name, 
        num_samples=Config.num_samples, 
        init_states=Config.init_states, 
        init_transitions=Config.init_transitions, 
        max_states=Config.max_states, 
        min_states=Config.min_states, 
        state_interval=Config.state_interval, 
        max_transitions=Config.max_transitions, 
        min_transitions=Config.min_transitions, 
        transition_interval=Config.transition_interval,
        density_interval=Config.density_interval,
        reduce_states=Config.reduce_states
    )
