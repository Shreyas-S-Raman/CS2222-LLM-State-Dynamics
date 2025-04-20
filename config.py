from sequence_generator import RandomLetterSequenceGenerator, DFAStateActionSequenceGenerator, DFAStateSequenceGenerator, DFAPDDLSequenceGenerator
import os
from dotenv import load_dotenv

load_dotenv()
hf_key = os.getenv("HF_KEY")
os.environ["HF_TOKEN"] = hf_key

class Config:

    model_name = 'pythia-70m'

    BASE_PATH = os.path.relpath(f'./{model_name}')
    
    if not os.path.exists(BASE_PATH):
        os.mkdir(BASE_PATH)

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
    model_name = 'pythia-14m'