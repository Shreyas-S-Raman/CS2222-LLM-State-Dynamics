import os
from dotenv import load_dotenv

load_dotenv()
hf_key = os.getenv("HF_KEY")
os.environ["HF_TOKEN"] = hf_key

class Config:
    #base arguments overriden in evaluate_model.py
    reduce_states = False
    model_name = 'pythia-14m'
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
