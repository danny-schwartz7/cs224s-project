# Inputs: ASR model, content sample, style sample
# Outputs: Mel spectrogram

from collections import Counter
import copy
import sys
sys.path.insert(0, 'asr')

import wandb

STYLE_TRANSFER_ARCHIVE = 'embeddings_cache'

from klepto.archives import dir_archive

import torch
from torch import Tensor, nn
from torch import optim
import torch.utils.data

import tqdm

import dataset

from pytorch_lightning.loggers import WandbLogger

WANDB_NAME = 'anivegesana'


from typing import NamedTuple
class ItemClass(NamedTuple): # copied from dataset.py
  input_feature: torch.Tensor
  input_mean: float
  input_std: float
  input_length: int
  input_path: str
  human_transcript_label: torch.Tensor
  human_transcript_length: int
  speaker_idx: int


class SampleCache:
    embeddings_mat: torch.Tensor  # a style_embedding_dim x (S + C + N) matrix
    # where C is the number of utterances with the same speaker as the content sample
    # S is the number of utterances with the target speaker
    # and N is the number of samples that will be in the negative sampling pool (all of them may not be used in a particular gradient update step)
    # the style speaker's embeddings will appear first among the columns of the embedding matrix.
    num_target_style_cols: int  # the number of embeddings from the target style speaker in embedding_mat
    num_content_style_samples: int # the number of samples from the content style

    content_speaker: int
    target_speaker: int

    def __init__(self, asr, content_speaker: int, target_speaker: int, num_negative_samples_per_speaker: int,
        num_negative_speakers: int, device, batch_size=8, seed=10):

        asr.eval()
        asr.droupout_enable = False
        asr.requires_grad_(False)

        self.content_speaker = content_speaker
        self.target_speaker = target_speaker
        self.num_negative_samples_per_speaker = num_negative_samples_per_speaker
        self.target_speaker = target_speaker

        self.content_samples = content_samples = asr.train_dataset.hf_ds.filter(
            is_speaker(content_speaker),
            # keep_in_memory=True
        )

        self.target_samples = target_samples = asr.train_dataset.hf_ds.filter(
            is_speaker(target_speaker),
            # keep_in_memory=True
        )

        S = len(target_samples)
        C = len(content_samples)
        N = num_negative_samples_per_speaker * num_negative_speakers
        num_negative_samples_needed = N

        negative_samples = []
        taken_per_speaker = Counter()

        for sample in tqdm.tqdm(asr.train_dataset.hf_ds.shuffle(seed)):
            speaker_id = sample['speaker_id']
            if speaker_id in (content_speaker, target_speaker):
                continue
            if taken_per_speaker[speaker_id] >= num_negative_samples_per_speaker:
                continue

            taken_per_speaker[speaker_id] += 1

            negative_samples.append(sample)

            num_negative_samples_needed -= 1
            if num_negative_samples_needed == 0:
                break

        all_samples = []
        all_samples.extend(content_samples)
        all_samples.extend(target_samples)
        all_samples.extend(negative_samples)

        self.sample_locs = [sample['audio']['path'] for sample in all_samples]

        all_samples_dataset = copy.copy(asr.train_dataset)
        all_samples_dataset.hf_ds = all_samples
        all_samples_loader = torch.utils.data.DataLoader(all_samples_dataset, batch_size)

        self.embeddings_mat = torch.zeros((64, S+C+N))
        self.num_target_style_cols = S
        self.num_content_style_samples = C

        for i, batch in enumerate(tqdm.tqdm(all_samples_loader)):
            batch = batch.my_to(device)
            self.embeddings_mat[:, i*batch_size:(i+1)*batch_size] = asr.get_style_embedding(batch, split='train').T
            torch.cuda.empty_cache()

        self.embeddings_mat /= torch.linalg.norm(self.embeddings_mat, dim=0, keepdim=True)


def is_speaker(speaker):
    def f(sample):
        return sample['speaker_id'] == speaker
    return f


class ListDataset(torch.utils.data.IterableDataset):
    def __init__(self, l):
        self.l = l
    def __iter__(self):
        return iter(self.l)
    def __len__(self):
        return len(self.l)
    def __getitem__(self, i):
        return self.l[i]


def contrastive_style_loss(output_style_embedding: torch.Tensor, cache: SampleCache, num_negative_samples: int) -> torch.Tensor:
    mag = torch.linalg.norm(output_style_embedding)
    output_style_embedding = output_style_embedding/mag
    numerator = torch.sum(output_style_embedding.T @ cache.embeddings_mat[:, :cache.num_target_style_cols])

    num_mandatory_samples = cache.num_target_style_cols + cache.num_content_style_samples

    # populate part of denominator without random negative sampling
    denominator = numerator + torch.sum(output_style_embedding.T @ cache.embeddings_mat[:, cache.num_target_style_cols:num_mandatory_samples])

    neg_sample_weights = torch.ones((cache.embeddings_mat.shape[1] - num_mandatory_samples),)
    neg_samples = torch.multinomial(neg_sample_weights, num_negative_samples)  # can use 'generator' parameter for reproducible randomness
    neg_sample_indices = neg_samples + num_mandatory_samples

    neg_sample_embeddings_mat = cache.embeddings_mat[:, neg_sample_indices]

    # populate rest of denominator with neg samples
    denominator = denominator + torch.sum(output_style_embedding.T @ neg_sample_embeddings_mat)

    loss = -torch.log(numerator) + torch.log(denominator)

    return loss


class StyleTransfer:
    def __init__(self, asr, content_sample: ItemClass, sample_cache: SampleCache, loss_func, num_negative_samples: int):
        self.asr = asr
        self.content_sample = content_sample
        self.sample_cache = sample_cache
        self.loss_func = loss_func
        self.num_negative_samples = num_negative_samples

        wandb.init(project='cs224s-style_transfer', entity=WANDB_NAME, name=f'{sample_cache.content_speaker}:{sample_cache.target_speaker}', 
                    config={}, sync_tensorboard=True)
        self.wandb_logger = WandbLogger()

    def optimize(self, ignore_content_loss: bool=True, max_steps=200, patience=5):
        '''
        Modifies self.content_sample in place
        '''
        prev_loss = float('inf')
        best_loss = float('inf')

        output_style_embedding = copy.deepcopy(self.content_sample.input_feature).cuda()

        optimizer = optim.Adam(params=
            [output_style_embedding]
        )
        output_style_embedding.requires_grad_(True)
        self.asr.train()
        self.asr.droupout_enable = False
        self.asr.requires_grad_(False)

        step = 0
        iters_since_best = 0

        while iters_since_best < patience and step < max_steps:
            optimizer.zero_grad()

            loss = self.loss_func(output_style_embedding, self.sample_cache, self.num_negative_samples)

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
        content_mean = self.content_sample.input_mean
        content_std = self.content_sample.input_std
        pretransfer_utterance = get_utterance(self.asr, self.content_sample)
        self.optimize()
        posttransfer_utterance = get_utterance(self.asr, self.content_sample)

        content_mels = content_mels[0, :self.content_sample.input_length.long().item(), :]
        posttransfer_mels = self.content_sample.input_feature[0, :self.content_sample.input_length.long().item(), :]

        wandb.finish()

        return {
            'pretransfer_utterance': pretransfer_utterance,
            'posttransfer_utterance': posttransfer_utterance,
            'content_mels': content_mels,
            'posttransfer_mels': posttransfer_mels,
            'content_mean': content_mean,
            'content_std': content_std,
        }





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
