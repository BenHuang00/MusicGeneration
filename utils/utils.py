import os
import json
import pickle
import psutil
import GPUtil


def make_file_path(path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)


def _check_file_path(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'File not found: {file_path}')


def _save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def _load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def _save_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def _load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def _save_txt(data, file_path):
    with open(file_path, 'w') as f:
        f.write(data)


def _load_txt(file_path):
    with open(file_path, 'r') as f:
        return f.read()


def save_file(data, file_path, lock=False):
    make_file_path(file_path)
    file_ext = os.path.splitext(file_path)[1]
    if file_ext == '.json':
        _save_json(data, file_path)
    elif file_ext == '.pkl':
        _save_pickle(data, file_path)
    elif file_ext == '.txt':
        _save_txt(data, file_path)
    else:
        raise ValueError(f'Unsupported file extension: {file_ext}')

    if lock:
        os.chmod(file_path, 0o444)  # read-only


def load_file(file_path):
    _check_file_path(file_path)
    file_ext = os.path.splitext(file_path)[1]
    if file_ext == '.json':
        return _load_json(file_path)
    elif file_ext == '.pkl':
        return _load_pickle(file_path)
    elif file_ext == '.txt':
        return _load_txt(file_path)
    else:
        raise ValueError(f'Unsupported file extension: {file_ext}')

def get_system_info():
    print(f'[!] System Information:')
    print(f'[!]      General Information:')
    print(f'[!]            OS: {os.name}')
    print(f'[!]      CPU Information:')
    print(f'[!]            Physical cores: {psutil.cpu_count(logical=False)}')
    print(f'[!]            Logical cores: {psutil.cpu_count(logical=True)}')
    print(f'[!]            Max CPU Frequency: {psutil.cpu_freq().max:.2f}Mhz')
    print(f'[!]            Min CPU Frequency: {psutil.cpu_freq().min:.2f}Mhz')
    print(f'[!]      Memory Information:')
    print(f'[!]            Total Memory: {psutil.virtual_memory().total / 1024**3:.2f}GB')

    gpus_info = GPUtil.getGPUs()
    for gpu_info in gpus_info:
        print(f'[!]      GPU Information:')
        print(f'[!]            Name: {gpu_info.name}')
        print(f'[!]            Memory: {gpu_info.memoryTotal:.0f}MB')
        print(f'[!]            Max GPU Frequency: {gpu_info.maxFrequency:.2f}Mhz')
        print(f'[!]            Min GPU Frequency: {gpu_info.minFrequency:.2f}Mhz')
