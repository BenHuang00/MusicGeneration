import os
import argparse

from tqdm import tqdm

import config
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
    print(f'Saved tokens2ids: {len(token2id)} tokens')

    save_file(id2token, ids2tokens_path)
    print(f'Saved ids2tokens: {len(id2token)} tokens')

    return token2id, id2token


def preprocess_dataset(metadata, tokens2ids):
    split_data = []
    not_found_list = []
    for key, value in tqdm(metadata.items(), desc='Preprocessing dataset'):
        token_path = os.path.join(args.dataset_path, value['tokens.txt'])
        try:
            tokens = load_file(token_path).split()
            if not args.exclude_file_head:
                tokens = tokens[:-1]
            else:
                tokens = tokens[4:-1]
            for i in range(0, len(tokens) - args.window_size, args.window_step):
                window = tokens[i:i + args.window_size]
                window_ids = [tokens2ids[token] for token in window]
                target = tokens[i + args.window_size]
                target_id = tokens2ids[target]
                split_data.append((window_ids, target_id))
        except FileNotFoundError:
            not_found_list.append(token_path)
    print(f'Not found: {not_found_list}')
    print(f'Preprocessed dataset: {len(split_data)} windows')
    dataset = GPDataset(split_data)
    save_file(dataset, os.path.join(args.preprocess_path, config.preprocess_dataset_name))
    print(f'Saved dataset: {len(dataset)} windows')


def preprocess():
    dataset_all_metadata_path = os.path.join(args.dataset_path, config.dataset_all_metadata_name)
    dataset_all_tokens_path = os.path.join(args.dataset_path, config.dataset_all_tokens_name)

    tokens2ids_path = os.path.join(args.preprocess_path, config.tokens2ids_name)
    ids2tokens_path = os.path.join(args.preprocess_path, config.ids2tokens_name)

    print('Preprocessing dataset...')

    metadata = load_file(dataset_all_metadata_path)
    print(f'Loaded metadata: {len(metadata)} songs')

    tokens = load_file(dataset_all_tokens_path)
    print(f'Loaded tokens: {len(tokens)} tokens')

    tokens2ids, ids2tokens = preprocess_tokens2ids(tokens, tokens2ids_path, ids2tokens_path)

    print('Preprocessing dataset...')
    preprocess_dataset(metadata, tokens2ids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess dataset')

    parser.add_argument('--dataset_path', type=str, default=config.dataset_path, help='Path to the dataset')
    parser.add_argument('--preprocess_path', type=str, default=config.preprocess_path,
                        help='Path to save the preprocessed files')

    parser.add_argument('--window_size', type=int, default=config.window_size, help='Size of the window')
    parser.add_argument('--window_step', type=int, default=config.window_step, help='Step of the window')

    parser.add_argument('--exclude_file_head', action='store_true', default=config.exclude_file_head,
                        help='Exclude the file head')

    args = parser.parse_args()

    preprocess()
