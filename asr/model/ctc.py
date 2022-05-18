from dataset import LibriDatasetAdapter, get_cer_per_sample
import datasets
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import * # type: ignore

__all__ = ('get_ctc_loss', 'CTCMetricsDict', 'CTCEncoderDecoder', 'LightningCTC')

class CTCMetricsDict:
  class Val(TypedDict):
    val_loss: float
    val_cer: float
  class Test(TypedDict):
    test_loss: float
    test_cer: float

def get_ctc_loss(
    log_probs, targets, input_lengths, target_lengths, blank=0):
  """Connectionist Temporal Classification objective function."""
  ctc_loss = None
  log_probs = log_probs.contiguous()
  targets = targets.long()
  input_lengths = input_lengths.long()
  target_lengths = target_lengths.long()
  ############################ START OF YOUR CODE ############################
  # TODO(2.1)
  # Hint:
  # - `F.ctc_loss`: https://pytorch.org/docs/stable/nn.functional.html#ctc-loss
  # - log_probs is passed in with shape (batch_size, input_length, num_classes).
  # - Notice that `F.ctc_loss` expects log_probs of shape
  #   (input_length, batch_size, num_classes)
  # - Turn on zero_infinity.
  
  ctc_loss = torch.mean(
      F.ctc_loss(log_probs.transpose(0, 1), targets, input_lengths, \
      target_lengths, blank=blank, reduction='none', zero_infinity=True)
  )
  
  ############################# END OF YOUR CODE #############################
  return ctc_loss

class CTCEncoderDecoder(nn.Module):
  """
  Encoder-Decoder model trained with CTC objective.

  Args:
    input_dim: integer
                number of input features
    num_class: integer
                size of transcription vocabulary
    num_layers: integer (default: 2)
                number of layers in encoder LSTM
    hidden_dim: integer (default: 128)
                number of hidden dimensions for encoder LSTM
    bidirectional: boolean (default: True)
                    is the encoder LSTM bidirectional?
  """
  def __init__(
      self, input_dim, num_class, num_layers=2, hidden_dim=128,
      bidirectional=True):
    super().__init__()
    # Note: `batch_first=True` argument implies the inputs to the LSTM should
    # be of shape (batch_size x T x D) instead of (T x batch_size x D).
    self.dropout_enable = True
    self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                            bidirectional=bidirectional, batch_first=True)
    self.decoder = nn.Linear(hidden_dim * 2, num_class)
    self.input_dim = input_dim
    self.num_class = num_class
    self.num_layers = num_layers
    self.hidden_dim = hidden_dim
    self.embedding_dim = hidden_dim * num_layers * 2 * \
                          (2 if bidirectional else 1)

  def combine_h_and_c(self, h, c):
    """Combine the signals from RNN hidden and cell states."""
    batch_size = h.size(1)
    h = h.permute(1, 0, 2).contiguous()
    c = c.permute(1, 0, 2).contiguous()
    h = h.view(batch_size, -1)
    c = c.view(batch_size, -1)
    return torch.cat([h, c], dim=1)  # just concatenate

  def forward(self, inputs, input_lengths):
    batch_size, max_length, _ = inputs.size()
    # `torch.nn.utils.rnn.pack_padded_sequence` collapses padded sequences
    # to a contiguous chunk
    inputs = torch.nn.utils.rnn.pack_padded_sequence(
        inputs, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
    log_probs = None
    h, c = None, None
    ############################ START OF YOUR CODE ############################
    # TODO(2.1)
    # Hint:
    # - Refer to https://pytorch.org/docs/stable/nn.html
    # - Use `self.encoder` to get the encodings output which is of shape
    #   (batch_size, max_length, num_directions*hidden_dim) and the
    #   hidden states and cell states which are both of shape
    #   (batch_size, num_layers*num_directions, hidden_dim)
    # - Pad outputs with `0.` using `torch.nn.utils.rnn.pad_packed_sequence`
    #   (turn on batch_first and set total_length as max_length).
    # - Apply 50% dropout.
    # - Use `self.decoder` to take the embeddings sequence and return
    #   probabilities for each character.
    # - Make sure to then convert to log probabilities.

    encodings, (h, c) = self.encoder(inputs)
    encodings, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(
        encodings, batch_first=True, total_length=max_length)
    if self.dropout_enable:
      encodings = F.dropout(encodings, 0.5, training=self.training)
    vals = self.decoder(encodings)
    log_probs = F.log_softmax(vals, dim=-1)

    ############################# END OF YOUR CODE #############################
    
    # The extracted embedding is not used for the ASR task but will be
    # needed for other auxiliary tasks.
    embedding = self.combine_h_and_c(h, c)
    return log_probs, embedding

  def get_loss(
      self, log_probs, targets, input_lengths, target_lengths, blank=0):
    return get_ctc_loss(
        log_probs, targets, input_lengths, target_lengths, blank)

  def decode(self, log_probs, input_lengths, labels, label_lengths,
             sos_index, eos_index, pad_index, eps_index):
    # Use greedy decoding.
    decoded = torch.argmax(log_probs, dim=2)
    batch_size = decoded.size(0)
    # Collapse each decoded sequence using CTC rules.
    hypotheses = []
    for i in range(batch_size):
      hypotheses_i = self.ctc_collapse(decoded[i], input_lengths[i].item(),
                                       blank_index=eps_index)
      hypotheses.append(hypotheses_i)

    hypothesis_lengths = input_lengths.cpu().numpy().tolist()
    if labels is None: # Run at inference time.
      references, reference_lengths = None, None
    else:
      references = labels.cpu().numpy().tolist()
      reference_lengths = label_lengths.cpu().numpy().tolist()

    return hypotheses, hypothesis_lengths, references, reference_lengths

  def ctc_collapse(self, seq, seq_len, blank_index=0):
    result = []
    for i, tok in enumerate(seq[:seq_len]):
      if tok.item() != blank_index:  # remove blanks
        if i != 0 and tok.item() == seq[i-1].item():  # remove dups
          pass
        else:
          result.append(tok.item())
    return result


class LightningCTC(pl.LightningModule):
  """PyTorch Lightning class for training a CTC model.

  Args:
    n_mels: number of mel frequencies. (default: 128)          
    n_fft: number of fourier features. (default: 256)          
    win_length: number of frames in a window. (default: 256)              
    hop_length: number of frames to hop in computing spectrogram. (default: 128)               
    wav_max_length: max number of timesteps in a waveform spectrogram. (default: 200)                  
    transcript_max_length: max number of characters in decoded transcription. (default: 200)                         
    learning_rate: learning rate for Adam optimizer. (default: 1e-3)                  
    batch_size: batch size used in optimization and evaluation. (default: 256)               
    weight_decay: weight decay for Adam optimizer. (default: 1e-5)               
    encoder_num_layers: number of layers in LSTM encoder. (default: 2)                       
    encoder_hidden_dim: number of hidden dimensions in LSTM encoder. (default: 256)
    encoder_bidirectional: directionality of LSTM encoder. (default: True)                         
  """
  def __init__(self, n_mels=128, n_fft=256, win_length=256, hop_length=128, 
               wav_max_length=200, transcript_max_length=200, 
               learning_rate=1e-3, batch_size=256, weight_decay=1e-5, 
               encoder_num_layers=2, encoder_hidden_dim=256,
               fmin=0, fmax=8000, sr=22050, 
               encoder_bidirectional=True):
    super().__init__()
    self.save_hyperparameters()
    self.n_mels = n_mels
    self.n_fft = n_fft
    self.win_length = win_length
    self.hop_length = hop_length
    self.lr = learning_rate
    self.batch_size = batch_size
    self.weight_decay = weight_decay
    self.wav_max_length = wav_max_length
    self.transcript_max_length = transcript_max_length
    self.fmin = fmin
    self.fmax = fmax
    self.sr = sr
    self.train_dataset, self.val_dataset, self.test_dataset = \
      self.create_datasets()
    self.encoder_num_layers = encoder_num_layers
    self.encoder_hidden_dim = encoder_hidden_dim
    self.encoder_bidirectional = encoder_bidirectional

    # Instantiate the CTC encoder/decoder.
    self.model = self.create_model()

  def create_model(self):
    model = CTCEncoderDecoder(
      self.train_dataset.input_dim,
      self.train_dataset.num_class,
      num_layers=self.encoder_num_layers,
      hidden_dim=self.encoder_hidden_dim,
      bidirectional=self.encoder_bidirectional)
    return model

  @property
  def dropout_enable(self):
    return self.model.dropout_enable
  
  @dropout_enable.setter
  def dropout_enable(self, b):
    self.model.dropout_enable = b

  def create_datasets(self):
    train_dataset = LibriDatasetAdapter(
        datasets.load_dataset('librispeech_asr', 'clean', split='train.100'), n_mels=self.n_mels, n_fft=self.n_fft, # type: ignore
        win_length=self.win_length, hop_length=self.hop_length,
        wav_max_length=self.wav_max_length,
        transcript_max_length=self.transcript_max_length,
        fmin=self.fmin, fmax=self.fmax, sr=self.sr,
        append_eos_token=False)
    val_dataset = LibriDatasetAdapter(
        datasets.load_dataset('librispeech_asr', 'clean', split='validation'), n_mels=self.n_mels, n_fft=self.n_fft, # type: ignore
        win_length=self.win_length, hop_length=self.hop_length, 
        wav_max_length=self.wav_max_length,
        transcript_max_length=self.transcript_max_length,
        fmin=self.fmin, fmax=self.fmax, sr=self.sr,
        append_eos_token=False) 
    test_dataset = LibriDatasetAdapter(
        datasets.load_dataset('librispeech_asr', 'clean', split='test'), n_mels=self.n_mels, n_fft=self.n_fft, # type: ignore
        win_length=self.win_length, hop_length=self.hop_length,
        wav_max_length=self.wav_max_length,
        transcript_max_length=self.transcript_max_length,
        fmin=self.fmin, fmax=self.fmax, sr=self.sr,
        append_eos_token=False) 
    return train_dataset, val_dataset, test_dataset

  def configure_optimizers(self):
    optim = torch.optim.AdamW(self.model.parameters(),
                              lr=self.lr, weight_decay=self.weight_decay)
    return [optim], [] # <-- put scheduler in here if you want to use one

  def get_loss(self, log_probs, input_lengths, labels, label_lengths):
    loss = self.model.get_loss(log_probs, labels, input_lengths, label_lengths,
                                blank=self.train_dataset.eps_index)
    return loss

  def forward(self, inputs, input_lengths, labels, label_lengths):
    log_probs, embedding = self.model(inputs, input_lengths)
    return log_probs, embedding

  def get_primary_task_loss(self, batch, split='train'):
    """Returns ASR model losses, metrics, and embeddings for a batch."""
    inputs, input_lengths = batch.input_feature, batch.input_length
    labels, label_lengths = batch.human_transcript_label, batch.human_transcript_length

    if split == 'train':
      log_probs, embedding = self.forward(
          inputs, input_lengths, labels, label_lengths)
    else:
      # do not pass labels to not teacher force after training
      log_probs, embedding = self.forward(
          inputs, input_lengths, None, None)

    loss = self.get_loss(log_probs, input_lengths, labels, label_lengths)

    # Compute CER (no gradient necessary).
    with torch.no_grad():
      hypotheses, hypothesis_lengths, references, reference_lengths = \
        self.model.decode(
            log_probs, input_lengths, labels, label_lengths,
            self.train_dataset.sos_index,
            self.train_dataset.eos_index,
            self.train_dataset.pad_index,
            self.train_dataset.eps_index)
      cer_per_sample = get_cer_per_sample(
          hypotheses, hypothesis_lengths, references, reference_lengths)
      cer = cer_per_sample.mean()
      metrics = {f'{split}_loss': loss, f'{split}_cer': cer}

    return loss, metrics, embedding

  # Overwrite TRAIN
  def training_step(self, batch, batch_idx):
    loss, metrics, _ = self.get_primary_task_loss(batch, split='train')
    # self.log_dict(metrics)
    self.log('train_loss', loss, prog_bar=True, on_step=True)
    self.log('train_cer', metrics['train_cer'], prog_bar=True, on_step=True)
    return loss

  # Overwrite VALIDATION: get next minibatch
  def validation_step(self, batch, batch_idx):
    loss, metrics, _ = self.get_primary_task_loss(batch, split='val')
    return metrics

  def test_step(self, batch, batch_idx):
    _, metrics, _ = self.get_primary_task_loss(batch, split='test')
    return metrics

  # Overwrite: e.g. accumulate stats (avg over CER and loss)
  def validation_epoch_end(self, outputs: list[CTCMetricsDict.Val]):
    """Called at the end of validation step to aggregate outputs."""
    # outputs is list of metrics from every validation_step (over a
    # validation epoch).
    metrics = { 
      # important that these are torch Tensors!
      'val_loss': torch.tensor([elem['val_loss']
                                for elem in outputs]).float().mean(),
      'val_cer': torch.tensor([elem['val_cer']
                                for elem in outputs]).float().mean()
    }
    self.log('val_loss', metrics['val_loss'], prog_bar=True)
    self.log('val_cer', metrics['val_cer'], prog_bar=True)
    # self.log_dict(metrics)

  def test_epoch_end(self, outputs: list[CTCMetricsDict.Test]):
    metrics = { 
      'test_loss': torch.tensor([elem['test_loss']
                                  for elem in outputs]).float().mean(),
      'test_cer': torch.tensor([elem['test_cer']
                                for elem in outputs]).float().mean()
    }
    self.log_dict(metrics)
    
  def train_dataloader(self):
    # - important to shuffle to not overfit!
    # - drop the last batch to preserve consistent batch sizes
    loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                        shuffle=True, pin_memory=True, drop_last=True, num_workers=4)
    return loader

  def val_dataloader(self):
    loader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                        shuffle=False, pin_memory=True, num_workers=4)
    return loader

  def test_dataloader(self):
    loader = DataLoader(self.test_dataset, batch_size=self.batch_size,
                        shuffle=False, pin_memory=True, num_workers=4)
    return loader
