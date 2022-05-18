from sklearn.metrics import f1_score
from itertools import chain

import datasets
import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F

from dataset import LibriDatasetAdapter
from .ctc import *
from .las import *

from Constants import EFFECTIVE_NUM_SPEAKERS

class SpeakerIdClassifier(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_dim=64):
        super().__init__()
        ############################ START OF YOUR CODE ############################
        # TODO(4.2)

        self.linear = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, n_classes)
        self.loss_func = torch.nn.NLLLoss()

        ############################# END OF YOUR CODE #############################

    def forward(self, inputs):
        log_probs = None
        ############################ START OF YOUR CODE ############################
        # TODO(4.2)
        # Hint: This is an N-way classification problem.

        x = self.linear(inputs)
        x = F.relu(x)
        x = self.linear2(x)
        log_probs = F.log_softmax(x, dim=-1)

        ############################# END OF YOUR CODE #############################
        return log_probs

    def get_loss(self, probs, targets):
        loss = None
        ############################ START OF YOUR CODE ############################
        # TODO(4.2)

        loss = self.loss_func(probs, targets.squeeze())

        ############################# END OF YOUR CODE #############################
        return loss

    def get_style_embedding(self, inputs):
        x = self.linear(inputs)
        # x = F.relu(x)  # TODO: would it make sense to apply the ReLU here?
        return x


class CTCDecoder(nn.Module):
  """
  This is a small decoder (just one linear layer) that takes 
  the listener embedding from LAS and imposes a CTC 
  objective on the decoding.

  NOTE: This is only to be used for the Joint CTC-Attention model.
  """
  def __init__(self, listener_hidden_dim, num_class, dropout=0.5):
    super().__init__()
    self.fc = nn.Linear(listener_hidden_dim, num_class)
    self.dropout = nn.Dropout(dropout)
    self.dropout_enable = True
    self.listener_hidden_dim = listener_hidden_dim
    self.num_class = num_class

  def forward(self, listener_outputs):
    batch_size, maxlen, _ = listener_outputs.size()
    listener_outputs = self.dropout(listener_outputs)
    logits = self.fc(listener_outputs)
    logits = logits.view(batch_size, maxlen, self.num_class)
    log_probs = F.log_softmax(logits, dim=2)
    return log_probs

  def get_loss(
      self, log_probs, input_lengths, labels, label_lengths, blank=0):
    return get_ctc_loss(
      log_probs, labels, input_lengths, label_lengths, blank)


class JointCTCAttention(LASEncoderDecoder):
    """Joint CTC and LAS model that optimizes the LAS objective but
    regularized by the conditional independence of a CTC decoder. One
    can interpret CTC as regularizer on LAS.
    """

    def __init__(
            self, input_dim, num_class, label_maxlen, listener_hidden_dim=128,
            listener_bidirectional=True, num_pyramid_layers=3, dropout=0,
            speller_hidden_dim=256, speller_num_layers=1, mlp_hidden_dim=128,
            multi_head=1, sos_index=0, sample_decode=False):
        super().__init__(
            input_dim,
            num_class,
            label_maxlen,
            listener_hidden_dim=listener_hidden_dim,
            listener_bidirectional=listener_bidirectional,
            num_pyramid_layers=num_pyramid_layers,
            dropout=dropout,
            speller_hidden_dim=speller_hidden_dim,
            speller_num_layers=speller_num_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            multi_head=multi_head,
            sos_index=sos_index,
            sample_decode=sample_decode,
        )
        self.ctc_decoder = CTCDecoder(listener_hidden_dim * 2, num_class)
        self.num_pyramid_layers = num_pyramid_layers
        self.embedding_dim = listener_hidden_dim * 4

    @property
    def dropout_enable(self):
      return self.ctc_decoder.dropout_enable
    
    @dropout_enable.setter
    def dropout_enable(self, b):
      self.ctc_decoder.dropout_enable = b
      
    def forward(
            self, inputs, ground_truth=None, teacher_force_prob=0.9, ):
        ctc_log_probs = None
        las_log_probs = None
        h, c = None, None
        ############################ START OF YOUR CODE ############################
        # TODO(5.1)
        # Hint:
        # - Encode the inputs with the `listener` network and decode
        #   transcription probabilities using both the `speller` network
        #   and CTCDecoder network.

        listener_outputs, (c, h) = self.listener(inputs)

        las_log_probs = self.speller(listener_outputs, ground_truth=ground_truth, teacher_force_prob=teacher_force_prob)

        ctc_input = listener_outputs
        ctc_log_probs = self.ctc_decoder(ctc_input)

        ############################# END OF YOUR CODE #############################
        listener_hc = self.combine_h_and_c(h, c)
        return ctc_log_probs, las_log_probs, listener_hc

    def get_loss(
            self, ctc_log_probs, las_log_probs, input_lengths, labels, label_lengths,
            num_labels, pad_index=0, blank_index=0, label_smooth=0.1):
        ctc_loss = self.ctc_decoder.get_loss(
            ctc_log_probs,
            # pyramid encode cuts timesteps in 1/2 each way
            input_lengths // (2 ** self.num_pyramid_layers),
            labels,
            label_lengths,
            blank=blank_index,
        )
        las_loss = super().get_loss(las_log_probs, labels, num_labels,
                                    pad_index=pad_index, label_smooth=label_smooth)

        return ctc_loss, las_loss

    def decode(self, log_probs, input_lengths, labels, label_lengths,
               sos_index, eos_index, pad_index, eps_index):
        las_log_probs = log_probs[1]
        return super().decode(las_log_probs, input_lengths, labels, label_lengths,
                              sos_index, eos_index, pad_index, eps_index)


class LightningCTCMTL(LightningCTC):
    """PyTorch Lightning class for training CTC with multi-task learning."""

    def __init__(self, n_mels=128, n_fft=256, win_length=256, hop_length=128,
                 wav_max_length=200, transcript_max_length=200,
                 learning_rate=1e-3, batch_size=256, weight_decay=1e-5,
                 encoder_num_layers=2, encoder_hidden_dim=256,
                 encoder_bidirectional=True, asr_weight=1.0, speaker_id_weight=1.0,
                 fmin=0, fmax=8000, sr=22050):
        super().__init__(
            n_mels=n_mels, hop_length=hop_length,
            wav_max_length=wav_max_length,
            transcript_max_length=transcript_max_length,
            learning_rate=learning_rate,
            batch_size=batch_size,
            weight_decay=weight_decay,
            encoder_num_layers=encoder_num_layers,
            encoder_hidden_dim=encoder_hidden_dim,
            encoder_bidirectional=encoder_bidirectional,
            fmin=fmin, fmax=fmax, sr=sr)
        self.save_hyperparameters()
        self.asr_weight = asr_weight
        self.speaker_id_weight = speaker_id_weight

        ############################ START OF YOUR CODE ############################
        # TODO(4.3)
        # Instantiate your auxiliary task models here.

        self.speaker_id_model = SpeakerIdClassifier(self.model.embedding_dim, EFFECTIVE_NUM_SPEAKERS)

        ############################# END OF YOUR CODE #############################

    def create_datasets(self):
        train_dataset = LibriDatasetAdapter(
            datasets.load_dataset('librispeech_asr', 'clean', split='train.100'),  # type: ignore
            n_mels=self.n_mels, n_fft=self.n_fft,
            win_length=self.win_length, hop_length=self.hop_length,
            wav_max_length=self.wav_max_length,
            transcript_max_length=self.transcript_max_length,
            fmin=self.fmin, fmax=self.fmax, sr=self.sr,
            append_eos_token=True)  # LAS adds a EOS token to the end of a sequence
        val_dataset = LibriDatasetAdapter(
            datasets.load_dataset('librispeech_asr', 'clean', split='validation'),  # type: ignore
            n_mels=self.n_mels, n_fft=self.n_fft,
            win_length=self.win_length, hop_length=self.hop_length,
            wav_max_length=self.wav_max_length,
            transcript_max_length=self.transcript_max_length,
            fmin=self.fmin, fmax=self.fmax, sr=self.sr,
            append_eos_token=True)
        test_dataset = LibriDatasetAdapter(
            datasets.load_dataset('librispeech_asr', 'clean', split='test'),  # type: ignore
            n_mels=self.n_mels, n_fft=self.n_fft,
            win_length=self.win_length, hop_length=self.hop_length,
            wav_max_length=self.wav_max_length,
            transcript_max_length=self.transcript_max_length,
            fmin=self.fmin, fmax=self.fmax, sr=self.sr,
            append_eos_token=True)
        return train_dataset, val_dataset, test_dataset

    def get_multi_task_loss(self, batch, split='train'):
        """Gets losses and metrics for all task heads."""
        # Compute loss on the primary ASR task.
        asr_loss, asr_metrics, embedding = self.get_primary_task_loss(batch, split)

        # Note: Not all of these have to be used (it is up to your design)
        speaker_id_log_probs = None
        ############################ START OF YOUR CODE ############################
        # TODO(4.3)
        # Implement multi-task learning by combining multiple objectives.
        # Define `combined_loss` here.

        # batch_len = len(batch)
        # task_type_labels, dialog_acts_labels, sentiment_labels = batch[batch_len - 3], batch[batch_len - 2], batch[
        #     batch_len - 1]

        speaker_id_labels = batch.speaker_idx

        speaker_id_log_probs = self.speaker_id_model(embedding)
        speaker_loss = self.speaker_id_model.get_loss(speaker_id_log_probs, speaker_id_labels)

        combined_loss = self.asr_weight * asr_loss + \
                        self.speaker_id_weight * speaker_loss

        ############################ END OF YOUR CODE ##############################

        with torch.no_grad():
            ############################ START OF YOUR CODE ##########################
            # TODO(4.3)
            # No additional code is required here. :)
            # We provide how to compute metrics for all possible auxiliary tasks and
            # store them in your metrics dictionary. Comment out the metrics for tasks
            # you do not plan to use.

            # TASK_TYPE: Compare predicted task type to true task type.
            speaker_id_preds = torch.argmax(speaker_id_log_probs, dim=1)
            speaker_id_acc = \
                (speaker_id_preds == speaker_id_labels).float().mean().item()

            metrics = {
                # Task losses.
                f'{split}_asr_loss': asr_metrics[f'{split}_loss'],
                f'{split}_speaker_id_loss': speaker_loss,
                # CER as ASR metric.
                f'{split}_asr_cer': asr_metrics[f'{split}_cer'],
                # Accuracy as sentiment metric.
                f'{split}_speaker_id_acc': speaker_id_acc,
            }
            ############################ END OF YOUR CODE ############################
        return combined_loss, metrics

    def configure_optimizers(self):
        parameters = chain(self.model.parameters(),
                           self.speaker_id_model.parameters())
        optim = torch.optim.AdamW(parameters, lr=self.lr,
                                  weight_decay=self.weight_decay)
        return [optim], []

    def training_step(self, batch, batch_idx):
        loss, metrics = self.get_multi_task_loss(batch, split='train')
        self.log_dict(metrics)
        # self.log('train_asr_loss', metrics['train_asr_loss'], prog_bar=True,
        #         on_step=True)
        # self.log('train_asr_cer', metrics['train_asr_cer'], prog_bar=True,
        #         on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self.get_multi_task_loss(batch, split='val')
        return metrics

    def validation_epoch_end(self, outputs):
        ############################ START OF YOUR CODE ############################
        # TODO(4.3)
        # No additional code is required here. :)
        # Comment out the metrics for tasks you do not plan to use.
        metrics = {
            'val_asr_loss': torch.tensor([elem['val_asr_loss']
                                          for elem in outputs]).float().mean(),
            'val_asr_cer': torch.tensor([elem['val_asr_cer']
                                         for elem in outputs]).float().mean(),
            'val_speaker_id_loss': torch.tensor([elem['val_speaker_id_loss']
                                                for elem in outputs]).float().mean(),
            'val_speaker_id_acc': torch.tensor([elem['val_speaker_id_acc']
                                               for elem in outputs]).float().mean(),
        }
        ############################# END OF YOUR CODE #############################
        # self.log('val_asr_loss', metrics['val_asr_loss'], prog_bar=True)
        # self.log('val_asr_cer', metrics['val_asr_cer'], prog_bar=True)
        self.log_dict(metrics)

    def test_step(self, batch, batch_idx):
        loss, metrics = self.get_multi_task_loss(batch, split='test')
        return metrics

    def test_epoch_end(self, outputs):
        ############################ START OF YOUR CODE ############################
        # TODO(4.3)
        # No additional code is required here. :)
        # Comment out the metrics for tasks you do not plan to use.
        metrics = {
            'test_asr_loss': torch.tensor([elem['test_asr_loss']
                                           for elem in outputs]).float().mean(),
            'test_asr_cer': torch.tensor([elem['test_asr_cer']
                                          for elem in outputs]).float().mean(),
            'test_speaker_id_loss': torch.tensor([elem['test_speaker_id_loss']
                                                 for elem in outputs]).float().mean(),
            'test_speaker_id_acc': torch.tensor([elem['test_speaker_id_acc']
                                                for elem in outputs]).float().mean(),
        }
        ############################# END OF YOUR CODE #############################
        # self.log('test_asr_loss', metrics['test_asr_loss'], prog_bar=True)
        # self.log('test_asr_cer', metrics['test_asr_cer'], prog_bar=True)
        self.log_dict(metrics)

    def get_style_embedding(self, batch, split='train'):
        asr_loss, asr_metrics, embedding = self.get_primary_task_loss(batch, split)
        return self.speaker_id_model(embedding)
    
class LightningLASMTL(LightningCTCMTL):
  """Train a Listen-Attend-Spell model along with the Multi-Task Objevtive.
  """

  def __init__(self, n_mels=128, n_fft=256, win_length=256, hop_length=128,
               wav_max_length=200, transcript_max_length=200,
               learning_rate=1e-3, batch_size=256, weight_decay=1e-5,
               encoder_num_layers=2, encoder_hidden_dim=256,
               encoder_bidirectional=True, encoder_dropout=0,
               decoder_hidden_dim=256, decoder_num_layers=1,
               decoder_multi_head=1, decoder_mlp_dim=128,
               asr_label_smooth=0.1, teacher_force_prob=0.9,
               asr_weight=1.0, speaker_id_weight=1.0,
               fmin=0, fmax=8000, sr=22050):
    self.encoder_dropout = encoder_dropout
    self.decoder_hidden_dim = decoder_hidden_dim
    self.decoder_num_layers = decoder_num_layers
    self.decoder_mlp_dim = decoder_mlp_dim
    self.decoder_multi_head = decoder_multi_head
    self.asr_label_smooth = asr_label_smooth
    self.teacher_force_prob = teacher_force_prob

    super().__init__(
      n_mels=n_mels, n_fft=n_fft,
      win_length=win_length, hop_length=hop_length,
      wav_max_length=wav_max_length,
      transcript_max_length=transcript_max_length,
      learning_rate=learning_rate,
      batch_size=batch_size,
      weight_decay=weight_decay,
      encoder_num_layers=encoder_num_layers,
      encoder_hidden_dim=encoder_hidden_dim,
      encoder_bidirectional=encoder_bidirectional,
      asr_weight=asr_weight,
      speaker_id_weight=speaker_id_weight,
      fmin=fmin, fmax=fmax, sr=sr)
    self.save_hyperparameters()

  def create_model(self):
    model = LASEncoderDecoder(
      self.train_dataset.input_dim,
      self.train_dataset.num_class,
      self.transcript_max_length,
      listener_hidden_dim=self.encoder_hidden_dim,
      listener_bidirectional=self.encoder_bidirectional,
      num_pyramid_layers=self.encoder_num_layers,
      dropout=self.encoder_dropout,
      speller_hidden_dim=self.decoder_hidden_dim,
      speller_num_layers=self.decoder_num_layers,
      mlp_hidden_dim=self.decoder_mlp_dim,
      multi_head=self.decoder_multi_head,
      sos_index=self.train_dataset.sos_index,
      sample_decode=False)
    return model

  def create_datasets(self):
      train_dataset = LibriDatasetAdapter(
          datasets.load_dataset('librispeech_asr', 'clean', split='train.100'), # type: ignore
          n_mels=self.n_mels, n_fft=self.n_fft,
          win_length=self.win_length, hop_length=self.hop_length,
          wav_max_length=self.wav_max_length,
          transcript_max_length=self.transcript_max_length,
          fmin=self.fmin, fmax=self.fmax, sr=self.sr,
          append_eos_token=True)  # LAS adds a EOS token to the end of a sequence
      val_dataset = LibriDatasetAdapter(
          datasets.load_dataset('librispeech_asr', 'clean', split='validation'), # type: ignore
          n_mels=self.n_mels, n_fft=self.n_fft,
          win_length=self.win_length, hop_length=self.hop_length,
          wav_max_length=self.wav_max_length,
          transcript_max_length=self.transcript_max_length,
          fmin=self.fmin, fmax=self.fmax, sr=self.sr,
          append_eos_token=True)
      test_dataset = LibriDatasetAdapter(
          datasets.load_dataset('librispeech_asr', 'clean', split='test'), # type: ignore
          n_mels=self.n_mels, n_fft=self.n_fft,
          win_length=self.win_length, hop_length=self.hop_length,
          wav_max_length=self.wav_max_length,
          transcript_max_length=self.transcript_max_length,
          fmin=self.fmin, fmax=self.fmax, sr=self.sr,
          append_eos_token=True)
      return train_dataset, val_dataset, test_dataset

  def forward(self, inputs, input_lengths, labels, label_lengths):
    log_probs, embedding = self.model(
      inputs,
      ground_truth=labels,
      teacher_force_prob=self.teacher_force_prob,
    )
    return log_probs, embedding

  def get_loss(self, log_probs, input_lengths, labels, label_lengths):
    loss = self.model.get_loss(log_probs, labels,
      self.train_dataset.num_labels,
      pad_index=self.train_dataset.pad_index,
      label_smooth=self.asr_label_smooth)
    return loss


class LightningCTCLASMTL(LightningLASMTL):

  def __init__(self, n_mels=128, n_fft=256, win_length=256, hop_length=128,
               wav_max_length=200, transcript_max_length=200,
               learning_rate=1e-3, batch_size=256, weight_decay=1e-5,
               encoder_num_layers=2, encoder_hidden_dim=256,
               encoder_bidirectional=True, encoder_dropout=0,
               decoder_hidden_dim=256, decoder_num_layers=1,
               decoder_multi_head=1, decoder_mlp_dim=128,
               asr_label_smooth=0.1, teacher_force_prob=0.9,
               ctc_weight=0.5, asr_weight=1.0, speaker_id_weight=1.0,
               fmin=0, fmax=8000, sr=22050):
    super().__init__(
      n_mels=n_mels,
      n_fft=n_fft,
      win_length=win_length,
      hop_length=hop_length,
      wav_max_length=wav_max_length,
      transcript_max_length=transcript_max_length,
      learning_rate=learning_rate,
      batch_size=batch_size,
      weight_decay=weight_decay,
      encoder_num_layers=encoder_num_layers,
      encoder_hidden_dim=encoder_hidden_dim,
      encoder_bidirectional=encoder_bidirectional,
      encoder_dropout=encoder_dropout,
      decoder_hidden_dim=decoder_hidden_dim,
      decoder_num_layers=decoder_num_layers,
      decoder_multi_head=decoder_multi_head,
      decoder_mlp_dim=decoder_mlp_dim,
      asr_label_smooth=asr_label_smooth,
      teacher_force_prob=teacher_force_prob,
      asr_weight=asr_weight,
      speaker_id_weight=speaker_id_weight,
      fmin=fmin, fmax=fmax, sr=sr)
    self.save_hyperparameters()
    self.ctc_weight = ctc_weight

  def create_model(self):
    model = JointCTCAttention(
      self.train_dataset.input_dim,
      self.train_dataset.num_class,
      self.transcript_max_length,
      listener_hidden_dim=self.encoder_hidden_dim,
      listener_bidirectional=self.encoder_bidirectional,
      num_pyramid_layers=self.encoder_num_layers,
      dropout=self.encoder_dropout,
      speller_hidden_dim=self.decoder_hidden_dim,
      speller_num_layers=self.decoder_num_layers,
      mlp_hidden_dim=self.decoder_mlp_dim,
      multi_head=self.decoder_multi_head,
      sos_index=self.train_dataset.sos_index,
      sample_decode=False)

    return model

  @property
  def dropout_enable(self):
    assert super().dropout_enable == self.model.dropout_enable
    return self.model.dropout_enable
  
  @dropout_enable.setter
  def dropout_enable(self, b):
    super().dropout_enable = b
    self.model.dropout_enable = b

  def forward(self, inputs, input_lengths, labels, label_lengths):
    ctc_log_probs, las_log_probs, embedding = self.model(
      inputs,
      ground_truth=labels,
      teacher_force_prob=self.teacher_force_prob)
    return (ctc_log_probs, las_log_probs), embedding

  def get_loss(self, log_probs, input_lengths, labels, label_lengths):
    (ctc_log_probs, las_log_probs) = log_probs
    ctc_loss, las_loss = self.model.get_loss(
      ctc_log_probs,
      las_log_probs,
      input_lengths,
      labels,
      label_lengths,
      self.train_dataset.num_labels,
      pad_index=self.train_dataset.pad_index,
      blank_index=self.train_dataset.eps_index,
      label_smooth=self.asr_label_smooth)
    loss = self.ctc_weight * ctc_loss + (1 - self.ctc_weight) * las_loss
    return loss
