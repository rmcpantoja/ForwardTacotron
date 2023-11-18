import os
import unittest
from pathlib import Path

import librosa
import numpy as np

from utils.dsp import DSP
from utils.files import read_config


test_path = os.path.dirname(os.path.abspath(__file__))
resource_path = Path(test_path) / 'tests/resources'
config = read_config(resource_path / 'test_config.yaml')
dsp = DSP.from_config(config)
#y = dsp.load_wav(resource_path/'test_wav.wav')
#mel = dsp.wav_to_mel(y)
mel = np.load(resource_path/'test_mel.npy')
print(f"Mel Lenght: {len(mel)}. Real lenght: {mel.shape[-1]}")
wav = dsp.griffinlim(mel)
dsp.save_wav(wav, resource_path / 'test_griffinlim2.wav')