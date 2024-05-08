import os
import json
import pickle


def _check_file_path(path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)


def _load_json(file_path):
    _check_file_path(file_path)
    with open(file_path, 'r') as f:
        return json.load(f)


def _save_json(data, file_path):
    _check_file_path(file_path)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def _save_pickle(data, file_path):
    _check_file_path(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def _load_pickle(file_path):
    _check_file_path(file_path)
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def load_file(file_path):
    _check_file_path(file_path)
    file_ext = os.path.splitext(file_path)[1]
    if file_ext == '.json':
        return _load_json(file_path)
    elif file_ext == '.pkl':
        return _load_pickle(file_path)
    else:
        raise ValueError(f'Unsupported file extension: {file_ext}')


def save_file(data, file_path, lock=False):
    _check_file_path(file_path)
    file_ext = os.path.splitext(file_path)[1]
    if file_ext == '.json':
        _save_json(data, file_path)
    elif file_ext == '.pkl':
        _save_pickle(data, file_path)
    else:
        raise ValueError(f'Unsupported file extension: {file_ext}')

    if lock:
        os.chmod(file_path, 0o444)  # read-only
