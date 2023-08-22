import os
import unittest
from pathlib import Path

import librosa
import numpy as np
import torch

from utils.dataset import tensor_to_ndarray
from utils.dsp import DSP
from utils.files import read_config
import torch.nn.functional as F


class TestDSP(unittest.TestCase):

    def setUp(self) -> None:
        test_path = os.path.dirname(os.path.abspath(__file__))
        self.resource_path = Path(test_path) / 'resources'
        config = read_config(self.resource_path / 'test_config.yaml')
        self.dsp = DSP.from_config(config)

    def load_wavs(self):
        wav_dir_path = self.resource_path / 'wavs'
        wav_files = [file.resolve() for file in wav_dir_path.iterdir() if file.suffix == '.wav']
        waveforms = [self.dsp.load_wav(file_path) for file_path in wav_files]
        waveforms = [waveform.to(self.dsp.device) for waveform in waveforms]
        return waveforms

    def test_melspectrogram(self) -> None:
        file = librosa.util.example('brahms')
        y = self.dsp.load_wav(file)[:, :10000]
        y = y.to(self.dsp.device)
        mel = self.dsp.waveform_to_mel(y)
        mel = tensor_to_ndarray(mel)
        expected = np.load(str(self.resource_path / 'test_mel.npy'))
        np.testing.assert_allclose(expected, mel, rtol=1e-5)

    def test_batched_wav_to_mel(self) -> None:
        # read wav files
        waveforms = self.load_wavs()

        # process in batch
        mels_batched = self.dsp.waveform_to_mel_batched(waveforms)

        # process one by one
        mels_single_processing = [self.dsp.waveform_to_mel(waveform) for waveform in waveforms]

        # compare results
        for mel_batched, mel_single in zip(mels_batched, mels_single_processing):
            mse = F.mse_loss(mel_batched, mel_single).item()
            self.assertLess(mse, 1e-10)

    def test_batched_volume_adjustment(self) -> None:
        # read wav files
        waveforms = self.load_wavs()

        target_dbfs = -30
        # process in batch
        normalized_batched = self.dsp.adjust_volume_batched(waveforms, target_dbfs=target_dbfs)

        # process one by one
        normalized_single_processing = [self.dsp.adjust_volume(waveform, target_dbfs=target_dbfs) for waveform in waveforms]

        # compare results
        for norm_batch, norm_single in zip(normalized_batched, normalized_single_processing):
            mse = F.mse_loss(norm_batch, norm_single).item()
            self.assertEqual(mse, 0)
