import os
from dotenv import load_dotenv

load_dotenv()
hf_key = os.getenv("HF_TOKEN")
os.environ["HF_TOKEN"] = hf_key

class Config:
    #base arguments overriden in evaluate_model.py
    reduce_states = True
    model_name = 'meta-llama/Llama-3.2-1B'
    BASE_PATH = None

    #num states range
    init_states = [1,2,3,4,5,6,7,8,9] if not reduce_states else [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    min_states = 10 if not reduce_states else 110
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

    #diff state same action transition evaluation
    diff_state_same_action_transitions = [6, 10, 14, 18, 22, 26, 30]
    diff_state_same_action_trials = 5

    # noop_transitions = [1, 5, 10, 20, 40, 50, 80, 100]
    noop_transitions = [1, 5, 10, 20, 40, 50]
    noop_trials = 1

    epsilon = 1e-20