from dataset import LibriDatasetAdapter
from datasets import load_dataset
import torch

from model import JointCTCAttention

if __name__ == "__main__":
    config = {
        'n_mels': 128,
        'n_fft': 512,
        'win_length': 512,
        'hop_length': 128,
        'wav_max_length': 4370 // 2,
        'transcript_max_length': 580,
        'learning_rate': 1e-3,
        'batch_size': 128,
        'weight_decay': 0,
        'encoder_num_layers': 3,
        'encoder_hidden_dim': 256,
        'encoder_bidirectional': True,
        'encoder_dropout': 0,
        'decoder_hidden_dim': 128,  # must be 2 x encoder_hidden_dim
        'decoder_num_layers': 2,
        'decoder_multi_head': 1,
        'decoder_mlp_dim': 64,
        'asr_label_smooth': 0.1,
        'teacher_force_prob': 0.9,
        'ctc_weight': 0.5,
        'asr_weight': 0.5,
        'speaker_id_weight': 0.5
    }

    test_dset = load_dataset("librispeech_asr", "clean", split="test")
    our_dset = LibriDatasetAdapter(test_dset, wav_max_length=4370//2 + 1)

    our_model = JointCTCAttention(
        our_dset.input_dim,
        our_dset.num_class,
        our_dset.transcript_max_length,
        listener_hidden_dim=config['encoder_hidden_dim'],
        listener_bidirectional=config['encoder_bidirectional'],
        num_pyramid_layers=config['encoder_num_layers'],
        dropout=config['encoder_dropout'],
        speller_hidden_dim=config['decoder_hidden_dim'],
        speller_num_layers=config['decoder_num_layers'],
        mlp_hidden_dim=config['decoder_mlp_dim'],
        multi_head=config['decoder_multi_head'],
        sos_index=our_dset.sos_index,
        sample_decode=False)

    loader = torch.utils.data.DataLoader(our_dset, batch_size=config['batch_size'],
                        shuffle=False, pin_memory=True, drop_last=True, num_workers=4)

    for item in loader:
        model_out = our_model(item.input_feature, item.human_transcript_label)

        raise ValueError("quitting because you don't want to get stuck in this loop")
