# Inputs: ASR model, content sample, style sample
# Outputs: Mel spectrogram

import copy
import sys
sys.path.insert(0, 'asr')

import wandb

STYLE_TRANSFER_ARCHIVE = 'style_transfers'

from klepto.archives import dir_archive

import torch
from torch import nn
from torch import optim
from dataset import ItemClass

from pytorch_lightning.loggers import WandbLogger

WANDB_NAME = 'anivegesana'

class StyleTransfer:
    def __init__(self, asr, content_sample: ItemClass, style_sample: ItemClass):
        self.asr = asr
        self.content_sample = content_sample
        self.style_sample = style_sample

        content_path = self.content_sample.input_path
        style_path = self.style_sample.input_path
        wandb.init(project='cs224s-style_transfer', entity=WANDB_NAME, name=content_path + ':' + style_path, 
                    config={}, sync_tensorboard=True)
        self.wandb_logger = WandbLogger()

    def optimize(self, ignore_content_loss: bool=True, max_steps=200, patience=5):
        '''
        Modifies self.content_sample in place
        '''
        prev_loss = float('inf')
        best_loss = float('inf')

        content_param = self.content_sample.input_feature
        #nn.parameter.Parameter(self.content_sample.input_feature, True)

        optimizer = optim.Adam(params=
            [content_param] # TODO: should we change the input length
        )
        self.content_sample.input_feature.requires_grad_(True)
        self.asr.train()
        self.asr.droupout_enable = False
        self.asr.requires_grad_(False)

        # Don't change. Don't need to recompute them over and over again.
        style_embedding = self.asr.get_style_embedding(self.style_sample, split='train')
        gram_style_embedding = gram_matrix(style_embedding)

        step = 0
        iters_since_best = 0

        while iters_since_best < patience and step < max_steps:
            optimizer.zero_grad()

            if ignore_content_loss:
                content_embedding : torch.Tensor = self.asr.get_style_embedding(self.content_sample, split='train')
                gram_content_embedding = gram_matrix(content_embedding)

                losses = torch.linalg.norm(gram_content_embedding - gram_style_embedding, ord='fro', dim=(1, 2))
                loss = losses.mean()
            else:
                raise NotImplementedError()

            loss.backward()
            optimizer.step()

            curr_loss = loss.item()
            if curr_loss < best_loss:
                iters_since_best = 0
                best_loss = curr_loss
            else:
                iters_since_best += 1

            step += 1
            self.wandb_logger.log_metrics({
                'style_transfer_loss': curr_loss,
                'loss_diff': prev_loss - curr_loss
            }, step)
            prev_loss = curr_loss

    def analyze(self):
        content_mels = copy.deepcopy(self.content_sample.input_feature)
        content_norm = self.content_sample.input_norm
        pretransfer_utterance = get_utterance(self.asr, self.content_sample)
        style_utterance = get_utterance(self.asr, self.style_sample)
        self.optimize()
        posttransfer_utterance = get_utterance(self.asr, self.content_sample)
        content_path = self.content_sample.input_path
        style_path = self.style_sample.input_path

        content_mels = content_mels[0, :self.content_sample.input_length.long().item(), :]
        style_mels = self.style_sample.input_feature[0, :self.style_sample.input_length.long().item(), :]
        posttransfer_mels = self.content_sample.input_feature[0, :self.content_sample.input_length.long().item(), :]

        d = {
            'content_path': content_path,
            'content_mels': content_mels,
            'content_norm': content_norm,
            'style_path': style_path,
            'style_mels': style_mels,
            'style_utterance': style_utterance,
            'pretransfer_utterance': pretransfer_utterance,
            'posttransfer_utterance': posttransfer_utterance,
            'posttransfer_mels': posttransfer_mels,
        }

        db = dir_archive(STYLE_TRANSFER_ARCHIVE, serialized=True, cached=False)
        db[(content_path, style_path)] = d

        wandb.finish()

        return d

def gram_matrix(a):
    b, h = a.shape
    return torch.bmm(a.reshape(b, h, 1), a.reshape(b, 1, h))

def get_utterance(asr, sample):
    log_probs, embedding = asr(
        sample.input_feature,
        sample.input_length,
        sample.human_transcript_label,
        sample.human_transcript_length
    )

    hypotheses, hypothesis_lengths, references, reference_lengths = \
        asr.model.decode(
            log_probs,
            sample.input_length,
            sample.human_transcript_label,
            sample.human_transcript_length,
        
            asr.train_dataset.sos_index,
            asr.train_dataset.eos_index,
            asr.train_dataset.pad_index,
            asr.train_dataset.eps_index)

    hypothesis = [int(c) for c in hypotheses[0]]
    hypothesis_chars = asr.train_dataset.indices_to_chars(hypothesis)
    utterance = ''.join(hypothesis_chars)

    return utterance
