from typing import Tuple, Dict, Any
import torch
from models.tacotron import Tacotron

def load_taco(checkpoint_path: str) -> Tuple[Tacotron, Dict[str, Any]]:
    print(f'Loading tts checkpoint {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    config = checkpoint['config']
    print(config)
    tts_model = Tacotron.from_config(config)
    tts_model.load_state_dict(checkpoint['model'])
    print(f'Loaded taco with step {tts_model.get_step()}')
    return tts_model, config

tts_model, config = load_taco("C:/Users/LENOVO_User/Downloads/latest_model.pt")