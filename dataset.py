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

def preprocess_libri(sample, append_eos_token=False, eos_index=1, num_special_tokens=4):
  initial = [
    VOCAB.index(ch) for ch in sample['text']
  ]
  processed_text = (np.array(initial) + num_special_tokens).tolist()
  if append_eos_token:
    processed_text.append(eos_index)
  sample['processed_text'] = processed_text
  return sample


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

class ItemClass(NamedTuple):
  input_feature: torch.Tensor
  input_length: int
  human_transcript_label: torch.Tensor
  human_transcript_length: int
  speaker_idx: int

class LibriDatasetAdapter(Dataset):
    def __init__(self, hf_ds: datasets.Dataset, n_mels=64, n_fft=256, win_length=256, # type: ignore 
            hop_length=128, wav_max_length=2192, transcript_max_length=580, # 576 is the max num of chars
            append_eos_token=False, fmin=125, fmax=7600, sr=22050):

        hf_ds = hf_ds.filter(lambda example: len(example['text'].split()) >= 5)
        self.wav_max_length = wav_max_length
        self.transcript_max_length = transcript_max_length

        self.input_dim = n_mels
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

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

        self.hf_ds = hf_ds.map(preprocess_libri, fn_kwargs={
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

        self.fmin = fmin
        self.fmax = fmax
        self.sr = sr

    def __getitem__(self, index):
        """Serves primary task data for a single utterance."""
        # TODO: Improve efficiency

        sample : ProcessedLibriSample = self.hf_ds[index] # type: ignore
        wav = sample['audio']['array']
        wav_sr = sample['audio']['sampling_rate']
        text = sample['processed_text']
        speaker_id = sample['speaker_id']

        input_feature, input_length = self.transform_wav(wav, wav_sr)
        human_transcript_label, human_transcript_length = self.transform_text(text)
        speaker_idx = SPEAKER_MAPPING[speaker_id]

        #return input_feature, input_length, human_transcript_label, human_transcript_length# TODO: , speaker_id
        return ItemClass(input_feature, input_length, human_transcript_label, human_transcript_length, speaker_idx)

    def transform_wav(self, wav, sr):
        wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sr)

        mel_feats = librosa.feature.melspectrogram(y=wav, sr=self.sr, 
            n_mels=self.n_mels, n_fft=self.n_fft,
            win_length=self.win_length, hop_length=self.hop_length,
            fmin=self.fmin, fmax=self.fmax)
        
        log_mel_feats = librosa.power_to_db(mel_feats)
        log_mel_feats = librosa.util.normalize(log_mel_feats)

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

'''
class HarperValleyBank(Dataset):
  """Dataset to be used to train CTC, LAS, and MTL.
  
  Args:
    root: string
          path to the data files.
    split: string (default: train)
            choices: train | val | test
            which split of data to load
    n_mels: integer (default: 128)
            number of mel frequencies
    n_fft: integer (default: 256)
            number of fourier components
    win_length: integer (default: 256)
                should be <= n_fft
    hop_length: integer (default: 128)
                number of frames to skip in between
    wav_max_length: integer (default: 4370)
                    maximum number of timesteps in a waveform
    transcript_max_length: integer (default: 200)
                            maximum number of timesteps in a transcript
    append_eos_token: boolean (default: False)
                      add EOS token to the end of every transcription
                      this is used for LAS (and LAS+CTC models)
  """
  def __init__(
      self, root, split='train', n_mels=128, n_fft=256, win_length=256, 
      hop_length=128, wav_max_length=4370, transcript_max_length=200, 
      append_eos_token=False):
    super().__init__()
    print(f'> Constructing HarperValleyBank {split} dataset...')

    self.label_data = np.load(os.path.join(root, 'labels.npz'))   
    self.root = root
    self.wav_max_length = wav_max_length
    self.transcript_max_length = transcript_max_length

    self.input_dim = n_mels
    self.n_mels = n_mels
    self.n_fft = n_fft
    self.win_length = win_length
    self.hop_length = hop_length

    # Prune away very short examples.
    # This returns a list of indices of examples longer than 3 words.
    valid_indices = prune_transcripts(self.label_data['human_transcripts'])

    # Decides which indices belong to which split.
    train_indices, val_indices, test_indices = self.split_data(valid_indices)

    if split == 'train':
      indices = train_indices
    elif split == 'val':
      indices = val_indices
    elif split == 'test':
      indices = test_indices
    else:
      raise Exception(f'Split {split} not supported.')

    raw_human_transcripts = self.label_data['human_transcripts'].tolist()
    human_transcript_labels = get_transcript_labels(
      raw_human_transcripts, VOCAB, SILENT_VOCAB)
  
    # Increment all indices by 4 to reserve the following special tokens:
    #   0 for epsilon
    #   1 for start-of-sentence (SOS)
    #   2 for end-of-sentence (EOS)
    #   3 for padding 
    num_special_tokens = 4
    human_transcript_labels = [list(np.array(lab) + num_special_tokens) 
                                for lab in human_transcript_labels]
    # CTC doesn't use SOS nor EOS; LAS doesn't use EPS but add anyway.
    eps_index, sos_index, eos_index, pad_index = 0, 1, 2, 3

    if append_eos_token:
      # Ensert an EOS token to the end of all the labels.
      # This is important for the LAS objective.
      human_transcript_labels_ = []
      for i in range(len(human_transcript_labels)):
        new_label_i = human_transcript_labels[i] + [eos_index]
        human_transcript_labels_.append(new_label_i)
      human_transcript_labels = human_transcript_labels_
    self.human_transcript_labels = human_transcript_labels
  
    # Include epsilon, SOS, and EOS tokens.
    self.num_class = len(VOCAB) + len(SILENT_VOCAB) + num_special_tokens
    self.num_labels = self.num_class  # These are interchangeable.
    self.eps_index = eps_index
    self.sos_index = sos_index
    self.eos_index = eos_index
    self.pad_index = pad_index # Use this index for padding.

    self.indices = indices

  def indices_to_chars(self, indices):
    # indices: list of integers in vocab
    # add special characters in front (since we did this above)
    full_vocab = ['<eps>', '<sos>', '<eos>', '<pad>'] + VOCAB + SILENT_VOCAB
    chars = [full_vocab[ind] for ind in indices]
    return chars

  def split_data(self, valid_indices, train_ratio = 0.8, val_ratio = 0.1):
    """Splits data into train, val, and test sets based on speaker. When 
    evaluating methods on the test split, we measure how well they generalize
    to new (unseen) speakers.
    
    Concretely, this stores and returns indices belonging to each split.
    """
    # Fix seed so everyone reproduces the same splits.
    rs = np.random.RandomState(42)

    speaker_ids = self.label_data['speaker_ids']
    unique_speaker_ids = sorted(list(set(speaker_ids)))
    unique_speaker_ids = np.array(unique_speaker_ids)

    # Shuffle so the speaker IDs are distributed.
    rs.shuffle(unique_speaker_ids)

    num_speaker = len(unique_speaker_ids)
    num_train = int(train_ratio * num_speaker)
    num_val = int(val_ratio * num_speaker)
    num_test = num_speaker - num_train - num_val

    train_speaker_ids = unique_speaker_ids[:num_train]
    val_speaker_ids = unique_speaker_ids[num_train:num_train+num_val]
    test_speaker_ids = unique_speaker_ids[num_train+num_val:]

    train_speaker_dict = dict(zip(train_speaker_ids, ['train'] * num_train))
    val_speaker_dict = dict(zip(val_speaker_ids, ['val'] * num_val))
    test_speaker_dict = dict(zip(test_speaker_ids, ['test'] * num_test))
    speaker_dict = {**train_speaker_dict, **val_speaker_dict, 
                    **test_speaker_dict} 

    train_indices, val_indices, test_indices = [], [], []
    for i in range(len(speaker_ids)):
      speaker_id = speaker_ids[i]
      if speaker_dict[speaker_id] == 'train':
          train_indices.append(i)
      elif speaker_dict[speaker_id] == 'val':
          val_indices.append(i)
      elif speaker_dict[speaker_id] == 'test':
          test_indices.append(i)
      else:
          raise Exception('split not recognized.')

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)

    # Make sure to only keep "valid indices" i.e. those with more than 4 
    # words in the transcription.
    train_indices = np.intersect1d(train_indices, valid_indices)
    val_indices = np.intersect1d(val_indices, valid_indices)
    test_indices = np.intersect1d(test_indices, valid_indices)

    return train_indices, val_indices, test_indices

  def get_primary_task_data(self, index):
    """Returns audio and transcript information for a single utterance.

    Args:
      index: Index of an utterance.

    Returns:
      log melspectrogram, wav length, transcript label, transcript length
    """
    input_feature = None
    input_length = None
    human_transcript_label = None
    human_transcript_length = None

    wav = self.waveform_data[f'{index}'][:] # An h5py file uses string keys.
    sr = 8000 # We fix the sample rate for you.

    ############################ START OF YOUR CODE ############################
    # TODO(1.1)
    # - Compute the mel spectrogram of the audio crop.
    # - Convert the mel spectrogram to log space and normalize it.
    # - This is your primary task feature. Note that models will expect feature
    #   inputs of shape (T, n_mels).
    # - Pad the feature so that all features are fixed-length and
    #   convert it into a tensor.
    # - Likewise, retrieve and pad the corresponding transcript label sequence.
    #
    # Hint:
    # - Refer to https://librosa.org/doc/latest/index.html.
    # - Use `librosa.feature.melspectrogram` and `librosa.util.normalize`.
    # - Make sure to use our provided sr, n_mels, n_fft, win_length, 
    # - and hop_length
    # - utils.py has helpful padding functions.

    mel_feats = librosa.feature.melspectrogram(y=wav, sr=sr, 
       n_mels=self.n_mels, fmax=4096, n_fft=self.n_fft,
       win_length=self.win_length, hop_length=self.hop_length)
    
    log_mel_feats = librosa.power_to_db(mel_feats)
    log_mel_feats = librosa.util.normalize(log_mel_feats)

    input_feature = log_mel_feats.T

    input_feature, input_length = pad_wav(input_feature, self.wav_max_length)
    
    label = self.human_transcript_labels[index]
    human_transcript_label, human_transcript_length = \
        pad_transcript_label(label, self.transcript_max_length, pad=self.pad_index)

    input_feature = torch.as_tensor(input_feature, dtype=torch.float32)
    human_transcript_label = torch.as_tensor(human_transcript_label, dtype=torch.float32)

    ############################# END OF YOUR CODE #############################

    return input_feature, input_length, human_transcript_label, human_transcript_length

  def load_waveforms(self):
    # Make a file pointer to waveforms file.
    waveform_h5 = h5py.File(os.path.join(self.root, 'data.h5'), 'r')
    self.waveform_data = waveform_h5.get('waveforms')

  def __getitem__(self, index):
    """Serves primary task data for a single utterance."""
    if not hasattr(self, 'waveform_data'):
      # Do this in __getitem__ function so we enable multiprocessing.
      self.load_waveforms()
    index = int(self.indices[index])
    return self.get_primary_task_data(index)

  def __len__(self):
    """Returns total number of utterances in the dataset."""
    return len(self.indices)
'''
