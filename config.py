import os
import torch


is_development = False


# dataset settings
dataset_path = 'datasets/DadaGP-v1.1' if not is_development else 'datasets/DadaGP-v1.1-dev'


# preprocess settings
window_size = 200
window_step = window_size

exclude_file_head = False

preprocess_path = 'datasets/preprocess' if not is_development else 'datasets/preprocess-dev'


# train settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_size_scale = 0.8
val_size_scale = 0.1
test_size_scale = 1 - train_size_scale - val_size_scale

batch_size = 128

embedding_dim = 256
hidden_dim = 512
num_layers = 2

lr = 1e-4

epochs = 2

train_log_path = 'train_log' if not is_development else 'train_log-dev'
trained_model_path = os.path.join(train_log_path, 'trained_models')
train_history_path = os.path.join(train_log_path, 'train_history')


# DO NOT MODIFY THE CODE BELOW #
dataset_all_metadata_name = '_DadaGP_all_metadata.json'
dataset_all_tokens_name = '_DadaGP_all_tokens.json'

tokens2ids_name = 'tokens2ids.json'
ids2tokens_name = 'ids2tokens.json'

preprocess_dataset_name = 'preprocess_dataset.pkl'

trained_model_paras_name_template = '{}_model-{}.pth'           # model_name, epoch
trained_model_full_name_template = '{}_model-full-{}.pth'       # model_name, epoch
train_history_name_template = 'history-{}-{}.json'              # model_name, epoch
loss_plot_name_template = 'loss_plot-{}.png'                    # model_name

# CHECK IF THE CONFIGURATION IS CORRECT #
assert test_size_scale > 0, (f'Invalid train, val, test size scale,'
                             f'train_size_scale + val_size_scale must be less than 1, '
                             f'that is, test_size_scale must be greater than 0\n'
                             f'\tGot: '
                             f'train_size={train_size_scale}, val_size={val_size_scale}, test_size={test_size_scale}')


if __name__ == '__main__':
    print('Config:')
    print(f'\tdevice: {device}')
    print(f'\tbatch_size: {batch_size}')
    print(f'\tembedding_dim: {embedding_dim}')
    print(f'\thidden_dim: {hidden_dim}')
    print(f'\tnum_layers: {num_layers}')
    print(f'\tlr: {lr}')
    print(f'\tepochs: {epochs}')
    print(f'\ttrain_size_scale: {train_size_scale}')
    print(f'\tval_size_scale: {val_size_scale}')
    print(f'\ttest_size_scale: {test_size_scale}')
    print(f'\twindow_size: {window_size}')
    print(f'\twindow_step: {window_step}')
    print(f'\texclude_file_head: {exclude_file_head}')
    print(f'\tdataset_path: {dataset_path}')
    print(f'\tpreprocess_path: {preprocess_path}')
    print(f'\ttrain_log_path: {train_log_path}')
    print(f'\ttrained_model_path: {trained_model_path}')
    print(f'\ttrain_history_path: {train_history_path}')
    print(f'\n')
    print(f'Config is correct')
