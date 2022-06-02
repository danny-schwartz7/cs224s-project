from model import *

import wandb
import random
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import natsort

import os
import glob

import numpy as np

MODEL_PATH = 'trained_models'

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

  dirpath = os.path.join(MODEL_PATH, ckpt_dir)
  checkpoint_callback = ModelCheckpoint(
    dirpath=dirpath,
    save_top_k=-1,
    verbose=True,
    monitor=monitor_key, 
    mode='min')
  
  wandb.init(project='cs224s', entity=WANDB_NAME, name=ckpt_dir, 
             config=config, sync_tensorboard=True)
  wandb_logger = WandbLogger()
  
  trainer_args = dict(
        max_epochs=epochs, min_epochs=epochs, enable_checkpointing=True,
        callbacks=checkpoint_callback, logger=wandb_logger
  )
  if resume:
    checkpoint_name = natsort.natsorted(glob.glob(os.path.join(dirpath, '*')))
    if len(checkpoint_name) > 0:
      checkpoint_name = checkpoint_name[-1]
      print("LOADED " + checkpoint_name)
      trainer_args['resume_from_checkpoint'] = checkpoint_callback.best_model_path = checkpoint_name

  if use_gpu:
    trainer = pl.Trainer(gpus=1, **trainer_args)
  else:
    trainer = pl.Trainer(**trainer_args)
  
  trainer.fit(system)
  result = trainer.test()
  return result


config = {
    'n_mels': 80, 
    'n_fft': 1024,
    'fmin': 0,
    'fmax': 8000,
    'sr': 22050,
    'win_length': 1024,
    'hop_length': 256,
    'wav_max_length': 3024,
    'transcript_max_length': 580,
    'learning_rate': 1e-5, #1e-3, 
    'batch_size': 12,
    'weight_decay': 0, 
    'encoder_num_layers': 2,
    'encoder_hidden_dim': 256//2,
    'encoder_bidirectional': True,
    'encoder_dropout': 0.2,
    'decoder_hidden_dim': 512//2,  # must be 2 x encoder_hidden_dim
    'decoder_num_layers': 2,
    'decoder_multi_head': 1,
    'decoder_mlp_dim': 64,
    'asr_label_smooth': 0.1,
    'teacher_force_prob': 0.9,
    'ctc_weight': 0.5,
    'asr_weight': 0.5,
    'speaker_id_weight': 0.5
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
if __name__ == '__main__':
  run(system="LightningCTCLASMTL", config=config, ckpt_dir='wavglow-styletransfer', epochs=300, use_gpu=True, resume=True, monitor_key='train_asr_cer')
