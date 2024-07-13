import os
import sys
import argparse
import yaml

from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import load_file, save_file

from dataset import GPDataset


def preprocess_tokens2ids(tokens, tokens2ids_path, ids2tokens_path):
    token2id = {}
    id2token = {}
    for token in tokens:
        if token not in token2id:
            token2id[token] = len(token2id)
            id2token[token2id[token]] = token

    save_file(token2id, tokens2ids_path)
    print(f'[+] Saved tokens2ids: {len(token2id)} tokens')

    save_file(id2token, ids2tokens_path)
    print(f'[+] Saved ids2tokens: {len(id2token)} tokens')

    return token2id, id2token


def preprocess_dataset(metadata, tokens2ids):
    split_data = []
    not_found_list = []
    for key, value in tqdm(metadata.items(), desc='Preprocessing dataset'):
        token_path = os.path.join(cfg.dataset_path, value['tokens.txt'])
        try:
            tokens = load_file(token_path).split()
            if not cfg.exclude_header:
                tokens = tokens[:-1]
            else:
                tokens = tokens[4:-1]
            for i in range(0, len(tokens) - cfg.window_size, cfg.window_step):
                window = tokens[i:i + cfg.window_size]
                window_ids = [tokens2ids[token] for token in window]
                target = tokens[i + cfg.window_size]
                target_id = tokens2ids[target]
                split_data.append((window_ids, target_id))
        except FileNotFoundError:
            not_found_list.append(token_path)
    print(f'[!] Not found: {not_found_list}')
    print(f'[+] Preprocessed dataset: {len(split_data)} windows')
    num_tokens = len(tokens2ids)
    dataset = GPDataset(split_data, num_tokens)
    save_file(dataset, os.path.join(cfg.preprocess_path, 'gpdataset.pkl'))
    print(f'[+] Saved dataset: {len(dataset)} windows')


def preprocess():
    dataset_all_metadata_path = os.path.join(cfg.dataset_path, '_DadaGP_all_metadata.json')
    dataset_all_tokens_path = os.path.join(cfg.dataset_path, '_DadaGP_all_tokens.json')

    tokens2ids_path = os.path.join(cfg.preprocess_path, 'tokens2ids.json')
    ids2tokens_path = os.path.join(cfg.preprocess_path, 'ids2tokens.json')

    metadata = load_file(dataset_all_metadata_path)
    print(f'[+] Loaded metadata: {len(metadata)} songs')

    tokens = load_file(dataset_all_tokens_path)
    print(f'[+] Loaded tokens: {len(tokens)} tokens')

    tokens2ids, ids2tokens = preprocess_tokens2ids(tokens, tokens2ids_path, ids2tokens_path)

    preprocess_dataset(metadata, tokens2ids)


def check_config():
    assert os.path.exists(cfg.dataset_path), f'Not found: {cfg.dataset_path}'
    assert cfg.window_size > 0, f'Invalid window size: {cfg.window_size}'
    assert cfg.window_step > 0, f'Invalid window step: {cfg.window_step}'
    assert cfg.exluce_header in [True, False], f'Invalid exclude_header: {cfg.exclude_header}'
    print('[+] Config checked')


if __name__ == '__main__':
    print('[+] Launch preprocess.py')

    default_config = yaml.full_load(open('preprocess/config.yaml', 'r'))

    parser = argparse.ArgumentParser(description='Preprocess dataset')

    parser.add_argument('--dataset_path', type=str, default=default_config['INFERENCE']['dataset_path'],
                        help='Path to the dataset')
    parser.add_argument('--preprocess_path', type=str, default=default_config['INFERENCE']['preprocess_path'],
                        help='Path to save the preprocessed dataset')

    parser.add_argument('--window_size', type=int, default=default_config['PREPROCESS']['window_size'],
                        help='Size of the window')
    parser.add_argument('--window_step', type=int, default=default_config['PREPROCESS']['window_step'],
                        help='Step of the window')
    parser.add_argument('--exclude_header', action='store_true', default=default_config['PREPROCESS']['exclude_header'],
                        help='Exclude header of the file')

    cfg = parser.parse_args()

    print('[!] Preprocess Configuration:')
    for key, value in vars(cfg).items():
        print(f'[!]\t{key}: {value}')

    check_config()

    preprocess()
