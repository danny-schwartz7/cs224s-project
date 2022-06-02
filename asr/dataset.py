from collections import OrderedDict

# import h5py
import math
import json
import torch
import numpy as np
from glob import glob
import librosa
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from typing import * # type: ignore
#from IPython.display import Audio
from Constants import SPEAKER_MAPPING

from utils import (
  prune_transcripts, pad_wav, pad_transcript_label, get_transcript_labels,
  get_cer_per_sample)

import datasets


# HarperValleyBank character vocabulary
import string
VOCAB = list(" '" + string.ascii_uppercase)

# 522320


class LibriSampleAudio(TypedDict):
  path: str
  array: np.ndarray
  sampling_rate: int

class LibriSample(TypedDict):
  file: str
  audio: dict
  text: str
  speaker_id: int
  chapter_id: int
  id: str

class ProcessedLibriSample(LibriSample):
  processed_text: torch.Tensor
  spec_mean: float
  spec_std: float

class ItemClass(NamedTuple):
  input_feature: torch.Tensor
  input_mean: float
  input_std: float
  input_length: int
  input_path: str
  human_transcript_label: torch.Tensor
  human_transcript_length: int
  speaker_idx: int

  @property
  def speaker_id(self):
    return self.speaker_idx

  def my_cuda(self):
    return ItemClass(
      self.input_feature.cuda(),
      self.input_mean.cuda() if hasattr(self.input_mean, 'cuda') else self.input_mean,
      self.input_std.cuda() if hasattr(self.input_mean, 'cuda') else self.input_std,
      self.input_length,
      self.input_path,
      self.human_transcript_label.cuda(),
      self.human_transcript_length,
      self.speaker_idx
    )

  def my_cpu(self):
    return ItemClass(
      self.input_feature.cpu(),
      self.input_mean.cpu() if hasattr(self.input_mean, 'cpu') else self.input_mean,
      self.input_std.cpu() if hasattr(self.input_mean, 'cpu') else self.input_std,
      self.input_length,
      self.input_path,
      self.human_transcript_label.cpu(),
      self.human_transcript_length,
      self.speaker_idx
    )

  def my_to(self, device):
    return ItemClass(
      self.input_feature.to(device),
      self.input_mean.to(device) if hasattr(self.input_mean, 'to') else self.input_mean,
      self.input_std.to(device) if hasattr(self.input_mean, 'to') else self.input_std,
      self.input_length,
      self.input_path,
      self.human_transcript_label.to(device),
      self.human_transcript_length,
      self.speaker_idx
    )

def _filter_min_words(example, min_length=5):
  return len(example['text'].split()) >= min_length

class LibriDatasetAdapter(Dataset):
    def __init__(self, hf_ds: datasets.Dataset, n_mels=64, n_fft=256, win_length=256, # type: ignore 
            hop_length=128, wav_max_length=2192, transcript_max_length=580, # 576 is the max num of chars
            append_eos_token=False, fmin=125, fmax=7600, sr=22050):

        hf_ds = hf_ds.filter(_filter_min_words)
        self.wav_max_length = wav_max_length
        self.transcript_max_length = transcript_max_length

        self.input_dim = n_mels
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        self.fmin = fmin
        self.fmax = fmax
        self.sr = sr

        # raw_human_transcripts = [
        #     sample['text'] for sample in self.hf_ds
        # ]

        # human_transcript_labels = get_transcript_labels(
        #     raw_human_transcripts, VOCAB, [])
    
        # Increment all indices by 4 to reserve the following special tokens:
        #   0 for epsilon
        #   1 for start-of-sentence (SOS)
        #   2 for end-of-sentence (EOS)
        #   3 for padding 
        num_special_tokens = 4
        # human_transcript_labels = [list(np.array(lab) + num_special_tokens) 
        #                             for lab in human_transcript_labels]


        # CTC doesn't use SOS nor EOS; LAS doesn't use EPS but add anyway.
        eps_index, sos_index, eos_index, pad_index = 0, 1, 2, 3

        # if append_eos_token:
        #     # Ensert an EOS token to the end of all the labels.
        #     # This is important for the LAS objective.
        #     human_transcript_labels_ = []
        #     for i in range(len(human_transcript_labels)):
        #         new_label_i = human_transcript_labels[i] + [eos_index]
        #         human_transcript_labels_.append(new_label_i)
        #     human_transcript_labels = human_transcript_labels_

        self.hf_ds = hf_ds.map(self._preprocess_libri, fn_kwargs={
          'append_eos_token': append_eos_token,
          'eos_index': eos_index,
          'num_special_tokens': num_special_tokens
        })
        # self.human_transcript_labels = human_transcript_labels
    
        # Include epsilon, SOS, and EOS tokens.
        self.num_class = len(VOCAB) + num_special_tokens
        self.num_labels = self.num_class  # These are interchangeable.
        self.eps_index = eps_index
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.pad_index = pad_index # Use this index for padding.

    def _preprocess_libri(self, sample, append_eos_token=False, eos_index=1, num_special_tokens=4):
        # Process text
        initial = [
            VOCAB.index(ch) for ch in sample['text']
        ]
        processed_text = (np.array(initial) + num_special_tokens).tolist()
        if append_eos_token:
            processed_text.append(eos_index)
        sample['processed_text'] = processed_text

        # Process wav to get mean and std
        wav = sample['audio']['array']
        wav_sr = sample['audio']['sampling_rate']

        # Copy from transform_wav
        wav = librosa.resample(wav, orig_sr=wav_sr, target_sr=self.sr)

        mel_feats = librosa.feature.melspectrogram(y=wav, sr=self.sr, 
            n_mels=self.n_mels, n_fft=self.n_fft,
            win_length=self.win_length, hop_length=self.hop_length,
            fmin=self.fmin, fmax=self.fmax)
        
        log_mel_feats = librosa.power_to_db(mel_feats)

        input_std, input_mean = torch.std_mean(torch.as_tensor(log_mel_feats))

        sample['spec_mean'] = input_mean.item()
        sample['spec_std'] = input_std.item()

        return sample

    def __getitem__(self, index):
        """Serves primary task data for a single utterance."""

        sample : ProcessedLibriSample = self.hf_ds[index] # type: ignore
        wav = sample['audio']['array']
        wav_sr = sample['audio']['sampling_rate']
        input_path = sample['audio']['path']
        text = sample['processed_text']
        speaker_id = sample['speaker_id']
        input_mean = sample['spec_mean']
        input_std = sample['spec_std']

        input_feature, input_length = self.transform_wav(wav, wav_sr)
        human_transcript_label, human_transcript_length = self.transform_text(text)
        speaker_idx = SPEAKER_MAPPING[speaker_id]

        return ItemClass(input_feature, input_mean, input_std, input_length, input_path, human_transcript_label, human_transcript_length, speaker_idx)

    def transform_wav(self, wav, sr):
        wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sr)

        mel_feats = librosa.feature.melspectrogram(y=wav, sr=self.sr, 
            n_mels=self.n_mels, n_fft=self.n_fft,
            win_length=self.win_length, hop_length=self.hop_length,
            fmin=self.fmin, fmax=self.fmax)
        
        log_mel_feats = librosa.power_to_db(mel_feats)

        input_feature = log_mel_feats.T

        input_feature, input_length = pad_wav(input_feature, self.wav_max_length)
        input_feature = torch.as_tensor(input_feature, dtype=torch.float32)
        
        return input_feature, input_length

    def transform_text(self, text):
        label = text
        human_transcript_label, human_transcript_length = \
            pad_transcript_label(label, self.transcript_max_length, pad=self.pad_index)

        human_transcript_label = torch.as_tensor(human_transcript_label, dtype=torch.float32)

        return human_transcript_label, human_transcript_length

    def __len__(self):
        """Returns total number of utterances in the dataset."""
        return len(self.hf_ds)

    def indices_to_chars(self, indices):
        # indices: list of integers in vocab
        # add special characters in front (since we did this above)
        full_vocab = ['<eps>', '<sos>', '<eos>', '<pad>'] + VOCAB
        chars = [full_vocab[ind] for ind in indices]
        return chars

