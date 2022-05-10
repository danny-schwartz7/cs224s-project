from dataset import *
import datasets
import os
import torch
from torch import nn
import torch.nn.functional as F

from Constants import EFFECTIVE_NUM_SPEAKERS

MODEL_PATH = 'trained_models'

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
    encodings = F.dropout(encodings, 0.50, training=self.training)
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

class SpeakerIdClassifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        ############################ START OF YOUR CODE ############################
        # TODO(4.2)

        self.linear = nn.Linear(input_dim, n_classes)
        self.loss_func = torch.nn.NLLLoss()

        ############################# END OF YOUR CODE #############################

    def forward(self, inputs):
        log_probs = None
        ############################ START OF YOUR CODE ############################
        # TODO(4.2)
        # Hint: This is an N-way classification problem.

        log_probs = F.log_softmax(self.linear(inputs), dim=-1)

        ############################# END OF YOUR CODE #############################
        return log_probs

    def get_loss(self, probs, targets):
        loss = None
        ############################ START OF YOUR CODE ############################
        # TODO(4.2)

        loss = self.loss_func(probs, targets.squeeze())

        ############################# END OF YOUR CODE #############################
        return loss


class LASEncoderDecoder(nn.Module):
    def __init__(
            self, input_dim, num_class, label_maxlen, listener_hidden_dim=128,
            listener_bidirectional=True, num_pyramid_layers=3, dropout=0,
            speller_hidden_dim=256, speller_num_layers=1, mlp_hidden_dim=128,
            multi_head=1, sos_index=0, sample_decode=False):
        super().__init__()
        # Encoder.
        self.listener = Listener(input_dim, listener_hidden_dim,
                                 num_pyramid_layers=num_pyramid_layers,
                                 dropout=dropout,
                                 bidirectional=listener_bidirectional)
        # Decoder.
        self.speller = Speller(num_class, label_maxlen, speller_hidden_dim,
                               listener_hidden_dim, mlp_hidden_dim,
                               num_layers=speller_num_layers,
                               multi_head=multi_head,
                               sos_index=sos_index,
                               sample_decode=sample_decode)
        self.embedding_dim = listener_hidden_dim * 4

    def combine_h_and_c(self, h, c):
        batch_size = h.size(1)
        h = h.permute(1, 0, 2).contiguous()
        c = c.permute(1, 0, 2).contiguous()
        h = h.view(batch_size, -1)
        c = c.view(batch_size, -1)
        return torch.cat([h, c], dim=1)

    def forward(
            self, inputs, ground_truth=None, teacher_force_prob=0.9):
        log_probs = None
        h, c = None, None
        # this is the main model connection for forward prop
        outputs, (h, c) = self.listener(inputs)
        log_probs = self.speller(outputs, ground_truth=ground_truth, teacher_force_prob=teacher_force_prob)

        combined_h_and_c = self.combine_h_and_c(h, c)
        return log_probs, combined_h_and_c

    def get_loss(
            self, log_probs, labels, num_labels, pad_index=0, label_smooth=0.1):
        batch_size = log_probs.size(0)
        labels_maxlen = labels.size(1)

        if label_smooth == 0.0:
            loss = F.nll_loss(log_probs.view(batch_size * labels_maxlen, -1),
                              labels.long().view(batch_size * labels_maxlen),
                              ignore_index=pad_index)
        else:
            # label_smooth_loss is the sample as F.nll_loss but with a temperature
            # parameter that makes the log probability distribution "sharper".
            loss = label_smooth_loss(log_probs, labels.float(), num_labels,
                                     smooth_param=label_smooth)
        return loss

    def decode(self, log_probs, input_lengths, labels, label_lengths,
               sos_index, eos_index, pad_index, eps_index):
        # Use greedy decoding.
        decoded = torch.argmax(log_probs, dim=2)
        batch_size = decoded.size(0)
        # Collapse each decoded sequence using CTC rules.
        hypotheses = []
        hypothesis_lengths = []
        references = []
        reference_lengths = []
        for i in range(batch_size):
            decoded_i = decoded[i]
            hypothesis_i = []
            for tok in decoded_i:
                if tok.item() == sos_index:
                    continue
                if tok.item() == pad_index:
                    continue
                if tok.item() == eos_index:
                    # once we reach an EOS token, we are done generating.
                    break
                hypothesis_i.append(tok.item())
            hypotheses.append(hypothesis_i)
            hypothesis_lengths.append(len(hypothesis_i))

            if labels is not None:
                label_i = labels[i]
                reference_i = [tok.item() for tok in labels[i]
                               if tok.item() != sos_index and
                               tok.item() != eos_index and
                               tok.item() != pad_index]
                references.append(reference_i)
                reference_lengths.append(len(reference_i))

        if labels is None:  # Run at inference time.
            references, reference_lengths = None, None

        return hypotheses, hypothesis_lengths, references, reference_lengths

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
               asr_weight=1.0, speaker_id_weight=1.0):
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
      speaker_id_weight=speaker_id_weight)
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
          datasets.load_dataset('librispeech_asr', 'clean', split='train.100'),
          n_mels=self.n_mels, n_fft=self.n_fft,
          win_length=self.win_length, hop_length=self.hop_length,
          wav_max_length=self.wav_max_length,
          transcript_max_length=self.transcript_max_length,
          append_eos_token=True)  # LAS adds a EOS token to the end of a sequence
      val_dataset = LibriDatasetAdapter(
          datasets.load_dataset('librispeech_asr', 'clean', split='validation'),
          n_mels=self.n_mels, n_fft=self.n_fft,
          win_length=self.win_length, hop_length=self.hop_length,
          wav_max_length=self.wav_max_length,
          transcript_max_length=self.transcript_max_length,
          append_eos_token=True)
      test_dataset = LibriDatasetAdapter(
          datasets.load_dataset('librispeech_asr', 'clean', split='test'),
          n_mels=self.n_mels, n_fft=self.n_fft,
          win_length=self.win_length, hop_length=self.hop_length,
          wav_max_length=self.wav_max_length,
          transcript_max_length=self.transcript_max_length,
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
               ctc_weight=0.5, asr_weight=1.0, speaker_id_weight=1.0):
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
      speaker_id_weight=speaker_id_weight)
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


from utils import (reshape_and_apply, label_smooth_loss)


class Listener(nn.Module):
    """Stacks 3 layers of PyramidLSTMLayers to reduce resolution 8 times.

    Args:
      input_dim: Number of input features.
      hidden_dim: Number of hidden features.
      num_pyramid_layers: Number of stacked lstm layers. (default: 3)
      dropout: Dropout probability. (default: 0)
    """

    def __init__(
            self, input_dim, hidden_dim, num_pyramid_layers=3, dropout=0.,
            bidirectional=True):
        super().__init__()
        self.rnn_layer0 = PyramidLSTMLayer(input_dim, hidden_dim, num_layers=1,
                                           bidirectional=True, dropout=dropout)
        for i in range(1, num_pyramid_layers):
            setattr(
                self,
                f'rnn_layer{i}',
                PyramidLSTMLayer(hidden_dim * 2, hidden_dim, num_layers=1,
                                 bidirectional=bidirectional, dropout=dropout),
            )

        self.num_pyramid_layers = num_pyramid_layers

    def forward(self, inputs):
        outputs, hiddens = self.rnn_layer0(inputs)
        for i in range(1, self.num_pyramid_layers):
            outputs, hiddens = getattr(self, f'rnn_layer{i}')(outputs)
        return outputs, hiddens


class PyramidLSTMLayer(nn.Module):
    """A Pyramid LSTM layer is a standard LSTM layer that halves the size
    of the input in its hidden embeddings.
    """

    def __init__(self, input_dim, hidden_dim, num_layers=1,
                 bidirectional=True, dropout=0.):
        super().__init__()
        self.rnn = nn.LSTM(input_dim * 2, hidden_dim, num_layers=num_layers,
                           bidirectional=bidirectional, dropout=dropout,
                           batch_first=True)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

    def forward(self, inputs):
        batch_size, maxlen, input_dim = inputs.size()

        # reduce time resolution?
        inputs = inputs.contiguous().view(batch_size, maxlen // 2, input_dim * 2)
        outputs, hiddens = self.rnn(inputs)
        return outputs, hiddens


class AttentionLayer(nn.Module):
    """Attention module that trains an MLP to get attention weights."""

    def __init__(self, input_dim, hidden_dim, multi_head=1):
        super().__init__()

        self.phi = nn.Linear(input_dim, hidden_dim * multi_head)
        self.psi = nn.Linear(input_dim, hidden_dim)

        if multi_head > 1:
            self.fc_reduce = nn.Linear(input_dim * multi_head, input_dim)

        self.multi_head = multi_head
        self.hidden_dim = hidden_dim

    def forward(self, decoder_state, listener_feat):
        attention_score = None
        context = None
        input_dim = listener_feat.size(2)
        # decoder_state: batch_size x 1 x decoder_hidden_dim
        # listener_feat: batch_size x maxlen x input_dim
        comp_decoder_state = F.relu(self.phi(decoder_state))
        comp_listener_feat = F.relu(reshape_and_apply(self.psi, listener_feat))

        if self.multi_head == 1:
            energy = torch.bmm(
                comp_decoder_state,
                comp_listener_feat.transpose(1, 2)
            ).squeeze(1)
            attention_score = [F.softmax(energy, dim=-1)]
            weights = attention_score[0].unsqueeze(2).repeat(1, 1, input_dim)
            context = torch.sum(listener_feat * weights, dim=1)
        else:
            attention_score = []
            for att_query in torch.split(comp_decoder_state, self.hidden_dim, dim=-1):
                score = torch.softmax(
                    torch.bmm(att_query,
                              comp_listener_feat.transpose(1, 2)).squeeze(dim=1),
                )
                attention_score.append(score)

            projected_src = []
            for att_s in attention_score:
                weights = att_s.unsqueeze(2).repeat(1, 1, input_dim)
                proj = torch.sum(listener_feat * weights, dim=1)
                projected_src.append(proj)

            context = self.fc_reduce(torch.cat(projected_src, dim=-1))

        # context is the entries of listener input weighted by attention
        return attention_score, context


class Speller(nn.Module):
    """Decoder that uses a LSTM with attention to convert a sequence of
    hidden embeddings to a sequence of probabilities for output classes.
    """

    def __init__(
            self, num_labels, label_maxlen, speller_hidden_dim,
            listener_hidden_dim, mlp_hidden_dim, num_layers=1, multi_head=1,
            sos_index=0, sample_decode=False):
        super().__init__()
        self.rnn = nn.LSTM(num_labels + speller_hidden_dim,
                           speller_hidden_dim, num_layers=num_layers,
                           batch_first=True)
        self.attention = AttentionLayer(listener_hidden_dim * 2, mlp_hidden_dim,
                                        multi_head=multi_head)
        self.fc_out = nn.Linear(speller_hidden_dim * 2, num_labels)
        self.num_labels = num_labels
        self.label_maxlen = label_maxlen
        self.sample_decode = sample_decode
        self.sos_index = sos_index

    def step(self, inputs, last_hiddens, listener_feats):
        outputs, cur_hiddens = self.rnn(inputs, last_hiddens)
        attention_score, context = self.attention(outputs, listener_feats)
        features = torch.cat((outputs.squeeze(1), context), dim=-1)
        logits = self.fc_out(features)
        log_probs = torch.log_softmax(logits, dim=-1)

        return log_probs, cur_hiddens, context, attention_score

    def forward(
            self, listener_feats, ground_truth=None, teacher_force_prob=0.9):
        device = listener_feats.device
        if ground_truth is None:
            teacher_force_prob = 0
        teacher_force = np.random.random_sample() < teacher_force_prob

        batch_size = listener_feats.size(0)
        with torch.no_grad():
            output_toks = torch.zeros((batch_size, 1, self.num_labels), device=device)
            output_toks[:, 0, self.sos_index] = 1

        rnn_inputs = torch.cat([output_toks, listener_feats[:, 0:1, :]], dim=-1)

        hidden_state = None
        log_probs_seq = []

        if (ground_truth is None) or (not teacher_force_prob):
            max_step = int(self.label_maxlen)
        else:
            max_step = int(ground_truth.size(1))

        for step in range(max_step):
            log_probs, hidden_state, context, _ = self.step(
                rnn_inputs, hidden_state, listener_feats)
            log_probs_seq.append(log_probs.unsqueeze(1))

            if teacher_force:
                gt_tok = ground_truth[:, step:step + 1].float()
                output_tok = torch.zeros_like(log_probs)
                for idx, i in enumerate(gt_tok):
                    output_tok[idx, int(i.item())] = 1
                output_tok = output_tok.unsqueeze(1)
            else:
                if self.sample_decode:
                    probs = torch.exp(log_probs)
                    sampled_tok = Categorical(probs).sample()
                else:  # Pick max probability (greedy decoding)
                    output_tok = torch.zeros_like(log_probs)
                    sampled_tok = log_probs.topk(1)[1]

                output_tok = torch.zeros_like(log_probs)
                for idx, i in enumerate(sampled_tok):
                    output_tok[idx, int(i.item())] = 1
                output_tok = output_tok.unsqueeze(1)

            rnn_inputs = torch.cat([output_tok, context.unsqueeze(1)], dim=-1)

        # batch_size x maxlen x num_labels
        log_probs_seq = torch.cat(log_probs_seq, dim=1)

        return log_probs_seq.contiguous()


class LASEncoderDecoder(nn.Module):
    def __init__(
            self, input_dim, num_class, label_maxlen, listener_hidden_dim=128,
            listener_bidirectional=True, num_pyramid_layers=3, dropout=0,
            speller_hidden_dim=256, speller_num_layers=1, mlp_hidden_dim=128,
            multi_head=1, sos_index=0, sample_decode=False):
        super().__init__()
        # Encoder.
        self.listener = Listener(input_dim, listener_hidden_dim,
                                 num_pyramid_layers=num_pyramid_layers,
                                 dropout=dropout,
                                 bidirectional=listener_bidirectional)
        # Decoder.
        self.speller = Speller(num_class, label_maxlen, speller_hidden_dim,
                               listener_hidden_dim, mlp_hidden_dim,
                               num_layers=speller_num_layers,
                               multi_head=multi_head,
                               sos_index=sos_index,
                               sample_decode=sample_decode)
        self.embedding_dim = listener_hidden_dim * 4

    def combine_h_and_c(self, h, c):
        batch_size = h.size(1)
        h = h.permute(1, 0, 2).contiguous()
        c = c.permute(1, 0, 2).contiguous()
        h = h.view(batch_size, -1)
        c = c.view(batch_size, -1)
        return torch.cat([h, c], dim=1)

    def forward(
            self, inputs, ground_truth=None, teacher_force_prob=0.9):
        log_probs = None
        h, c = None, None
        # this is the main model connection for forward prop
        outputs, (h, c) = self.listener(inputs)
        log_probs = self.speller(outputs, ground_truth=ground_truth, teacher_force_prob=teacher_force_prob)

        combined_h_and_c = self.combine_h_and_c(h, c)
        return log_probs, combined_h_and_c

    def get_loss(
            self, log_probs, labels, num_labels, pad_index=0, label_smooth=0.1):
        batch_size = log_probs.size(0)
        labels_maxlen = labels.size(1)

        if label_smooth == 0.0:
            loss = F.nll_loss(log_probs.view(batch_size * labels_maxlen, -1),
                              labels.long().view(batch_size * labels_maxlen),
                              ignore_index=pad_index)
        else:
            # label_smooth_loss is the sample as F.nll_loss but with a temperature
            # parameter that makes the log probability distribution "sharper".
            loss = label_smooth_loss(log_probs, labels.float(), num_labels,
                                     smooth_param=label_smooth)
        return loss

    def decode(self, log_probs, input_lengths, labels, label_lengths,
               sos_index, eos_index, pad_index, eps_index):
        # Use greedy decoding.
        decoded = torch.argmax(log_probs, dim=2)
        batch_size = decoded.size(0)
        # Collapse each decoded sequence using CTC rules.
        hypotheses = []
        hypothesis_lengths = []
        references = []
        reference_lengths = []
        for i in range(batch_size):
            decoded_i = decoded[i]
            hypothesis_i = []
            for tok in decoded_i:
                if tok.item() == sos_index:
                    continue
                if tok.item() == pad_index:
                    continue
                if tok.item() == eos_index:
                    # once we reach an EOS token, we are done generating.
                    break
                hypothesis_i.append(tok.item())
            hypotheses.append(hypothesis_i)
            hypothesis_lengths.append(len(hypothesis_i))

            if labels is not None:
                label_i = labels[i]
                reference_i = [tok.item() for tok in labels[i]
                               if tok.item() != sos_index and
                               tok.item() != eos_index and
                               tok.item() != pad_index]
                references.append(reference_i)
                reference_lengths.append(len(reference_i))

        if labels is None:  # Run at inference time.
            references, reference_lengths = None, None

        return hypotheses, hypothesis_lengths, references, reference_lengths


class LightningCTCMTL(LightningCTC):
    """PyTorch Lightning class for training CTC with multi-task learning."""

    def __init__(self, n_mels=128, n_fft=256, win_length=256, hop_length=128,
                 wav_max_length=200, transcript_max_length=200,
                 learning_rate=1e-3, batch_size=256, weight_decay=1e-5,
                 encoder_num_layers=2, encoder_hidden_dim=256,
                 encoder_bidirectional=True, asr_weight=1.0, speaker_id_weight=1.0:
        super().__init__(
            n_mels=n_mels, hop_length=hop_length,
            wav_max_length=wav_max_length,
            transcript_max_length=transcript_max_length,
            learning_rate=learning_rate,
            batch_size=batch_size,
            weight_decay=weight_decay,
            encoder_num_layers=encoder_num_layers,
            encoder_hidden_dim=encoder_hidden_dim,
            encoder_bidirectional=encoder_bidirectional)
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
            datasets.load_dataset('librispeech_asr', 'clean', split='train.100'),
            n_mels=self.n_mels, n_fft=self.n_fft,
            win_length=self.win_length, hop_length=self.hop_length,
            wav_max_length=self.wav_max_length,
            transcript_max_length=self.transcript_max_length,
            append_eos_token=True)  # LAS adds a EOS token to the end of a sequence
        val_dataset = LibriDatasetAdapter(
            datasets.load_dataset('librispeech_asr', 'clean', split='validation'),
            n_mels=self.n_mels, n_fft=self.n_fft,
            win_length=self.win_length, hop_length=self.hop_length,
            wav_max_length=self.wav_max_length,
            transcript_max_length=self.transcript_max_length,
            append_eos_token=True)
        test_dataset = LibriDatasetAdapter(
            datasets.load_dataset('librispeech_asr', 'clean', split='test'),
            n_mels=self.n_mels, n_fft=self.n_fft,
            win_length=self.win_length, hop_length=self.hop_length,
            wav_max_length=self.wav_max_length,
            transcript_max_length=self.transcript_max_length,
            append_eos_token=True)
        return train_dataset, val_dataset, test_dataset

    def get_multi_task_loss(self, batch, split='train'):
        """Gets losses and metrics for all task heads."""
        # Compute loss on the primary ASR task.
        asr_loss, asr_metrics, embedding = self.get_primary_task_loss(batch, split)

        # Note: Not all of these have to be used (it is up to your design)
        task_type_labels = None
        dialog_acts_labels = None
        sentiment_labels = None
        speaker_id_log_probs = None
        dialog_acts_probs = None
        sentiment_log_probs = None
        speaker_loss = None
        dialog_acts_loss = None
        sentiment_loss = None
        combined_loss = None
        ############################ START OF YOUR CODE ############################
        # TODO(4.3)
        # Implement multi-task learning by combining multiple objectives.
        # Define `combined_loss` here.

        batch_len = len(batch)
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
            task_type_preds = torch.argmax(speaker_id_log_probs, dim=1)
            task_type_acc = \
                (task_type_preds == task_type_labels).float().mean().item()

            # DIALOG_ACTS: Compare predicted dialog actions to true dialog actions.
            dialog_acts_preds = torch.round(dialog_acts_probs)
            dialog_acts_f1 = f1_score(dialog_acts_labels.cpu().numpy().reshape(-1),
                                      dialog_acts_preds.cpu().numpy().reshape(-1))

            # SENTIMENT: Compare largest predicted sentiment to largest true sentim
            sentiment_preds = torch.argmax(sentiment_log_probs, dim=1)
            sentiment_labels = torch.argmax(sentiment_labels, dim=1)
            sentiment_acc = \
                (sentiment_preds == sentiment_labels).float().mean().item()

            metrics = {
                # Task losses.
                f'{split}_asr_loss': asr_metrics[f'{split}_loss'],
                f'{split}_task_type_loss': speaker_loss,
                f'{split}_dialog_acts_loss': dialog_acts_loss,
                f'{split}_sentiment_loss': sentiment_loss,
                # CER as ASR metric.
                f'{split}_asr_cer': asr_metrics[f'{split}_cer'],
                # Accuracy as task_type metric.
                f'{split}_task_type_acc': task_type_acc,
                # F1 score as dialog_acts metric.
                f'{split}_dialog_acts_f1': dialog_acts_f1,
                # Accuracy as sentiment metric.
                f'{split}_sentiment_acc': sentiment_acc,
            }
            ############################ END OF YOUR CODE ############################
        return combined_loss, metrics

    def configure_optimizers(self):
        parameters = chain(self.model.parameters(),
                           self.speaker_id_model.parameters(),
                           self.dialog_acts_model.parameters(),
                           self.sentiment_model.parameters())
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
            'val_task_type_loss': torch.tensor([elem['val_task_type_loss']
                                                for elem in outputs]).float().mean(),
            'val_task_type_acc': torch.tensor([elem['val_task_type_acc']
                                               for elem in outputs]).float().mean(),
            'val_dialog_acts_loss': torch.tensor([
                elem['val_dialog_acts_loss'] for elem in outputs]).float().mean(),
            'val_dialog_acts_f1': torch.tensor([elem['val_dialog_acts_f1']
                                                for elem in outputs]).float().mean(),
            'val_sentiment_loss': torch.tensor([elem['val_sentiment_loss']
                                                for elem in outputs]).float().mean(),
            'val_sentiment_acc': torch.tensor([elem['val_sentiment_acc']
                                               for elem in outputs]).float().mean()
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
            'test_task_type_loss': torch.tensor([elem['test_task_type_loss']
                                                 for elem in outputs]).float().mean(),
            'test_task_type_acc': torch.tensor([elem['test_task_type_acc']
                                                for elem in outputs]).float().mean(),
            'test_dialog_acts_loss': torch.tensor([
                elem['test_dialog_acts_loss'] for elem in outputs]).float().mean(),
            'test_dialog_acts_f1': torch.tensor([elem['test_dialog_acts_f1']
                                                 for elem in outputs]).float().mean(),
            'test_sentiment_loss': torch.tensor([elem['test_sentiment_loss']
                                                 for elem in outputs]).float().mean(),
            'test_sentiment_acc': torch.tensor([elem['test_sentiment_acc']
                                                for elem in outputs]).float().mean()
        }
        ############################# END OF YOUR CODE #############################
        # self.log('test_asr_loss', metrics['test_asr_loss'], prog_bar=True)
        # self.log('test_asr_cer', metrics['test_asr_cer'], prog_bar=True)
        self.log_dict(metrics)

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
               asr_weight=1.0, task_type_weight=1.0,
               dialog_acts_weight=1.0, sentiment_weight=1.0):
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
      task_type_weight=task_type_weight,
      dialog_acts_weight=dialog_acts_weight,
      sentiment_weight=sentiment_weight)
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
      datasets.load_dataset('librispeech_asr', 'clean', split='train.100'),
      n_mels=self.n_mels, n_fft=self.n_fft,
      win_length=self.win_length, hop_length=self.hop_length,
      wav_max_length=self.wav_max_length,
      transcript_max_length=self.transcript_max_length,
      append_eos_token=True)  # LAS adds a EOS token to the end of a sequence
    val_dataset = LibriDatasetAdapter(
      datasets.load_dataset('librispeech_asr', 'clean', split='validation'),
      n_mels=self.n_mels, n_fft=self.n_fft,
      win_length=self.win_length, hop_length=self.hop_length,
      wav_max_length=self.wav_max_length,
      transcript_max_length=self.transcript_max_length,
      append_eos_token=True)
    test_dataset = LibriDatasetAdapter(
      datasets.load_dataset('librispeech_asr', 'clean', split='test'),
      n_mels=self.n_mels, n_fft=self.n_fft,
      win_length=self.win_length, hop_length=self.hop_length,
      wav_max_length=self.wav_max_length,
      transcript_max_length=self.transcript_max_length,
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

# Do not modify.

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

  def create_datasets(self):
    train_dataset = LibriDatasetAdapter(
        datasets.load_dataset('librispeech_asr', 'clean', split='train.100'), n_mels=self.n_mels, n_fft=self.n_fft, 
        win_length=self.win_length, hop_length=self.hop_length,
        wav_max_length=self.wav_max_length,
        transcript_max_length=self.transcript_max_length,
        append_eos_token=False)
    val_dataset = LibriDatasetAdapter(
        datasets.load_dataset('librispeech_asr', 'clean', split='validation'), n_mels=self.n_mels, n_fft=self.n_fft,
        win_length=self.win_length, hop_length=self.hop_length, 
        wav_max_length=self.wav_max_length,
        transcript_max_length=self.transcript_max_length,
        append_eos_token=False) 
    test_dataset = LibriDatasetAdapter(
        datasets.load_dataset('librispeech_asr', 'clean', split='test'), n_mels=self.n_mels, n_fft=self.n_fft,
        win_length=self.win_length, hop_length=self.hop_length,
        wav_max_length=self.wav_max_length,
        transcript_max_length=self.transcript_max_length,
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
  def validation_epoch_end(self, outputs):
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

  def test_epoch_end(self, outputs):
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
    encodings = F.dropout(encodings, 0.50, training=self.training)
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

WANDB_NAME = 'anivegesana' # Fill in your Weights & Biases ID here.

def run(system, config, ckpt_dir, epochs=1, monitor_key='val_loss', 
        use_gpu=False, seed=1337, resume=False):
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)

  SystemClass = globals()[system]
  system = SystemClass(**config)

  checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(MODEL_PATH, ckpt_dir),
    save_top_k=1,
    verbose=True,
    monitor=monitor_key, 
    mode='min')
  
  wandb.init(project='cs224s', entity=WANDB_NAME, name=ckpt_dir, 
             config=config, sync_tensorboard=True)
  wandb_logger = WandbLogger()
  
  trainer_args = dict(
        max_epochs=epochs, min_epochs=epochs, enable_checkpointing=True,
        #num_workers=4,
        callbacks=checkpoint_callback, logger=wandb_logger
  )
  if resume and checkpoint_callback.best_model_path:
      trainer_args['resume_from_checkpoint'] = checkpoint_callback.best_model_path

  if use_gpu:
    trainer = pl.Trainer(gpus=1, **trainer_args)
  else:
    trainer = pl.Trainer(**trainer_args)
  
  trainer.fit(system)
  result = trainer.test()
  return result

config = {
    'n_mels': 128, 
    'n_fft': 512,
    'win_length': 512,
    'hop_length': 128,
    'wav_max_length': 4370//2, 
    'transcript_max_length': 580,
    'learning_rate': 1e-3, 
    'batch_size': 128, 
    'weight_decay': 0, 
    'encoder_num_layers': 2, 
    'encoder_hidden_dim': 256, 
    'encoder_bidirectional': True,
    'teacher_force_prob': 0.9,

}

# NOTES:
# -----
# - PyTorch Lightning will run 2 steps of validation prior to the first 
#   epoch to sanity check that validation works (otherwise you 
#   might waste an epoch training and error).
# - The progress bar updates very slowly, the model is likely 
#   training even if it doesn't look like it is. 
# - Wandb will generate a URL for you where all the metrics will be logged.
# - Every validation loop, the best performing model is saved.
# - After training, the system will evaluate performance on the test set.
run(system="LightningCTCLASMTL", config=config, ckpt_dir='ctc', epochs=20, use_gpu=True, resume=True)