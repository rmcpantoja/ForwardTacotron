import warnings
from collections import Counter
from pathlib import Path
from random import Random
from typing import Tuple, List, Dict

import numpy as np
from tabulate import tabulate
from torch.utils.data import DataLoader

from utils.dataset import PreprocessingDataPoint, PreprocessingDataset, tensor_to_ndarray
from utils.display import simple_table
from utils.dsp import DSP
from utils.mel_processing import spectrogram_torch
from utils.text.recipes import read_metadata

warnings.simplefilter(action='ignore', category=FutureWarning)

import tqdm
import argparse
import traceback
from multiprocessing import cpu_count

import torch
from resemblyzer import VoiceEncoder
from resemblyzer import preprocess_wav as preprocess_resemblyzer

from pitch_extraction.pitch_extractor import new_pitch_extractor_from_config, PitchExtractor
from utils.files import get_files, read_config, pickle_binary
from utils.paths import Paths
from utils.text.cleaners import Cleaner

import copy
SPEAKER_EMB_DIM = 256  # fixed speaker dim from VoiceEncoder


def valid_n_workers(num):
    n = int(num)
    if n < 1:
        raise argparse.ArgumentTypeError('%r must be an integer greater than 0' % num)
    return n


class PreprocessingBatchCollator:

    def __init__(self,
                 dsp: DSP,
                 text_dict: Dict[str, str],
                 cleaner: Cleaner,
                 pitch_extractor: PitchExtractor) -> None:
        self.dsp = dsp
        self.text_dict = text_dict
        self.cleaner = cleaner
        self.pitch_extractor = pitch_extractor

    def __call__(self, batch: List[Tuple[str, Path]]) -> List[PreprocessingDataPoint]:
        batch_data_points = []

        for item_id, path in batch:
            try:
                y = self.dsp.load_wav(path)

                reference_wav = preprocess_resemblyzer(tensor_to_ndarray(y), source_sr=self.dsp.sample_rate)

                if self.dsp.should_trim_long_silences:
                    y = self.dsp.trim_long_silences(y)
                if self.dsp.should_trim_start_end_silence:
                    y = self.dsp.trim_silence(y)

                if y.shape[-1] == 0:
                    print(f'Skipping {item_id} because of the zero length')
                    continue

                peak = torch.abs(y).max()
                if self.dsp.should_peak_norm or peak > 1.0:
                    y /= peak
                    y = y * 0.95
                audio = spectrogram_torch(y, self.dsp.win_length,
                    self.dsp.sample_rate, self.dsp.hop_length, self.dsp.win_length,
                    center=False)
                audio = torch.squeeze(audio, 0)
                pitch = self.pitch_extractor(y).astype(np.float32)
                cleaned_text = self.cleaner(text_dict[item_id])

                dp = PreprocessingDataPoint(
                    item_id=item_id,
                    text=cleaned_text,
                    pitch=pitch,
                    reference_wav=reference_wav,
                    processed_wav=y,
                    audio=audio
                )

                batch_data_points.append(dp)
            except Exception as e:
                print(f'Error processing {item_id}: {e}')
                print(traceback.format_exc())
                continue

        return batch_data_points


parser = argparse.ArgumentParser(description='Dataset preprocessing')
parser.add_argument('--path', '-p', help='directly point to dataset')
parser.add_argument('--config', metavar='FILE', default='configs/singlespeaker.yaml',
                    help='The config containing all hyperparams.')
parser.add_argument('--metafile', '-m', default='metadata.csv',
                    help='name of the metafile in the dataset dir')
parser.add_argument('--batch_size', '-b', metavar='N', type=int,
                    default=32, help='Batch size for preprocessing')
parser.add_argument('--num_workers', '-w', metavar='N', type=valid_n_workers,
                    default=cpu_count()-1, help='The number of worker threads to use for preprocessing')
args = parser.parse_args()


if __name__ == '__main__':
    config = read_config(args.config)
    audio_format = config['preprocessing']['audio_format']
    audio_files = get_files(Path(args.path), audio_format)
    file_id_to_audio = {w.name.replace(audio_format, ''): w for w in audio_files}
    audio_ids = set(file_id_to_audio.keys())
    paths = Paths(config['data_path'], config['tts_model_id'])
    n_workers = max(1, args.num_workers)
    batch_size = args.batch_size

    print(f'\nFound {len(audio_files)} {audio_format} files in "{args.path}".')
    assert len(audio_files) > 0, f'Found no {audio_format} files in {args.path}, exiting.'

    print('Preparing metadata...')
    text_dict, speaker_dict_raw = read_metadata(path=Path(args.path),
                                                metafile=args.metafile,
                                                format=config['preprocessing']['metafile_format'],
                                                n_workers=n_workers)

    text_dict = {item_id: text for item_id, text in text_dict.items()
                 if item_id in audio_ids and len(text) > config['preprocessing']['min_text_len']}
    file_id_to_audio = {k: v for k, v in file_id_to_audio.items() if k in text_dict}
    speaker_dict = {item_id: speaker for item_id, speaker in speaker_dict_raw.items() if item_id in audio_ids}
    speaker_counts = Counter(speaker_dict.values())

    assert len(file_id_to_audio) > 0, f'No audio file is indexed in metadata, exiting. ' \
                                      f'Pease make sure the audio ids match the ids in the metadata. ' \
                                      f'\nAudio ids: {sorted(list(audio_ids))[:5]}... ' \
                                      f'\nText ids: {sorted(list(speaker_dict_raw.keys()))[:5]}...'

    print(f'Will use {len(file_id_to_audio)} {audio_format} files that are indexed in metafile.\n')
    table = [(speaker, count) for speaker, count in speaker_counts.most_common()]
    print(tabulate(table, headers=('speaker', 'count')))

    nval = config['preprocessing']['n_val']
    if nval > len(file_id_to_audio):
        nval = len(file_id_to_audio) // 5
        print(f'\nWARNING: Using nval={nval} since the preset nval exceeds number of training files.')

    file_id_audio_list = list(file_id_to_audio.items())
    successful_ids = set()

    dataset = []
    cleaned_texts = []
    cleaner = Cleaner.from_config(config)
    pitch_extractor = new_pitch_extractor_from_config(config)
    dsp = DSP.from_config(config)
    preprocessing_batch_collator = PreprocessingBatchCollator(dsp=dsp,
                                                              cleaner=cleaner,
                                                              text_dict=text_dict,
                                                              pitch_extractor=pitch_extractor)

    simple_table([
        ('Sample Rate', dsp.sample_rate),
        ('Hop Length', dsp.hop_length),
        ('CPU Usage', f'{n_workers}/{cpu_count()}'),
        ('Num Validation', nval),
        ('Pitch Extraction', config['preprocessing']['pitch_extractor'])
    ])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    voice_encoder = VoiceEncoder().to(device)

    # Prepare processing
    print('Running preprocessing...')
    tts_dataset = PreprocessingDataset(file_id_audio_list)
    dataloader = DataLoader(tts_dataset,
                            num_workers=n_workers,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=preprocessing_batch_collator)

    # Process the dataset
    for batch in tqdm.tqdm(dataloader):
        # calculate mel spectrograms in a batch
        processed_wavs = [dp.processed_wav for dp in batch]
        mels = dsp.waveform_to_mel_batched(processed_wavs)

        # iterate over batch to perform non-batched computations and writing results
        for index, dp in enumerate(batch):
            if dp is not None and dp.item_id in text_dict:
                try:
                    mel = tensor_to_ndarray(mels[index])
                    np.save(paths.mel / f'{dp.item_id}.npy', mel, allow_pickle=False)
                    np.save(paths.raw_pitch / f'{dp.item_id}.npy', dp.pitch, allow_pickle=False)
                    speck_path = paths.audio / f'{dp.item_id}.pt'
                    torch.save(dp.audio, speck_path)
                    emb = voice_encoder.embed_utterance(dp.reference_wav)
                    np.save(paths.speaker_emb / f'{dp.item_id}.npy', emb, allow_pickle=False)

                    dataset += [(dp.item_id, mel.shape[-1])]
                    cleaned_texts += [(dp.item_id, dp.text)]
                    successful_ids.add(dp.item_id)
                except Exception as e:
                    print(traceback.format_exc())

    # filter according to successfully preprocessed datapoints
    text_dict = {k: v for k, v in text_dict.items() if k in successful_ids}
    speaker_dict = {k: v for k, v in speaker_dict.items() if k in successful_ids}
    speaker_counts = Counter(speaker_dict.values())

    # create stratified train / val split
    dataset.sort()
    random = Random(42)
    random.shuffle(dataset)
    val_ratio = nval / len(dataset)
    desired_val_counts = {speaker: max(count * val_ratio, 1) for speaker, count in speaker_counts.most_common()}
    val_speaker_counts = Counter()
    train_dataset, val_dataset = [], []
    for file_id, mel_len in dataset:
        speaker = speaker_dict[file_id]
        if val_speaker_counts.get(speaker, 0) < desired_val_counts[speaker]:
            val_dataset.append((file_id, mel_len))
            val_speaker_counts.update([speaker])
        else:
            train_dataset.append((file_id, mel_len))

    # sort val dataset longest to shortest
    val_dataset.sort(key=lambda d: -d[1])
    text_dict = {id: text for id, text in cleaned_texts}
    pickle_binary(text_dict, paths.text_dict)
    pickle_binary(speaker_dict, paths.speaker_dict)
    pickle_binary(train_dataset, paths.train_dataset)
    pickle_binary(val_dataset, paths.val_dataset)

    print('Averaging speaker embeddings...')

    mean_speaker_embs = {speaker: np.zeros(SPEAKER_EMB_DIM, dtype=float) for speaker in speaker_dict.values()}
    for file_id, speaker in tqdm.tqdm(speaker_dict.items(), total=len(speaker_dict), smoothing=0.1):
        emb = np.load(paths.speaker_emb / f'{file_id}.npy')
        mean_speaker_embs[speaker] += emb
    for speaker, emb in mean_speaker_embs.items():
        emb = emb / speaker_counts[speaker]
        emb = emb / np.linalg.norm(emb, 2)
        np.save(paths.mean_speaker_emb / f'{speaker}.npy', emb, allow_pickle=False)

    print('\n\nCompleted. Ready to run "python train_tacotron.py". \n')
