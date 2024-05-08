import os
import argparse

from tqdm import tqdm

import config
from utils import load_file, save_file


def preprocess_tokens2ids():
    token2id = {}
    id2token = {}
    for token in tokens:
        if token not in token2id:
            token2id[token] = len(token2id)
            id2token[token2id[token]] = token

    save_file(token2id, tokens2ids_path, lock=True)
    print(f'Saved tokens2ids: {len(token2id)} tokens')

    save_file(id2token, ids2tokens_path, lock=True)
    print(f'Saved ids2tokens: {len(id2token)} tokens')

    return token2id, id2token


def preprocess_songs():
    global total_files, pending_items, total_items

    def save_items():
        global total_files, pending_items

        save_file(pending_items, f'{args.preprocess_path}/batch-{total_files}.pkl', lock=True)
        total_files += 1
        pending_items.clear()

    for key, value in tqdm(metadata.items(), desc='Preprocessing songs'):
        tokens_path = os.path.join(args.dataset_path, value['tokens.txt'])

        try:
            with open(tokens_path, 'r') as f:
                tokens_song = f.read().split()
        except FileNotFoundError:
            print(f'File not found: {tokens_path}')
            continue

        tokens_song = [tokens2ids[token] for token in tokens_song]

        for i in range(0, len(tokens_song) - args.window_size - 1, args.window_step):
            window = tokens_song[i:i + args.window_size]
            target = tokens_song[i + args.window_size]
            total_items += 1
            pending_items.append((window, target))
            if len(pending_items) >= args.preprocess_batch_size:
                save_items()

    if pending_items:
        save_items()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess dataset')

    parser.add_argument('--dataset_path', type=str, default=config.dataset_path, help='Path to the dataset')
    parser.add_argument('--preprocess_path', type=str, default=config.preprocess_path,
                        help='Path to save the preprocessed files')

    parser.add_argument('--window_size', type=int, default=config.window_size, help='Size of the window')
    parser.add_argument('--window_step', type=int, default=config.window_step, help='Step of the window')
    parser.add_argument('--preprocess_batch_size', type=int, default=config.preprocess_batch_size,
                        help='Number of items within each file')

    args = parser.parse_args()

    dataset_all_metadata_path = os.path.join(args.dataset_path, config.dataset_all_metadata_name)
    dataset_all_tokens_path = os.path.join(args.dataset_path, config.dataset_all_tokens_name)

    tokens2ids_path = os.path.join(args.preprocess_path, config.tokens2ids_name)
    ids2tokens_path = os.path.join(args.preprocess_path, config.ids2tokens_name)

    print('Preprocessing dataset...')

    metadata = load_file(dataset_all_metadata_path)
    print(f'Loaded metadata: {len(metadata)} songs')

    tokens = load_file(dataset_all_tokens_path)
    print(f'Loaded tokens: {len(tokens)} tokens')

    tokens2ids, ids2tokens = preprocess_tokens2ids()

    total_files = 0
    total_items = 0
    pending_items = []

    preprocess_songs()

    preprocess_config = {
        'window_size': args.window_size,
        'window_step': args.window_step,
        'precess_batch_size': args.preprocess_batch_size,
        'total_files': total_files,
        'total_items': total_items,
        'preprocess_tokens2ids_name': config.tokens2ids_name,
        'preprocess_ids2tokens_name': config.ids2tokens_name,
    }
    save_file(preprocess_config, f'{args.preprocess_path}/config.pkl', lock=True)
