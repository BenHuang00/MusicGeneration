import os
import sys

import yaml

import argparse
from tqdm import tqdm

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import load_file, save_file


def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device(cfg.device))
    return model


def temperature(logits, temperature):
    probs = torch.exp(logits / temperature) / torch.sum(torch.exp(logits / temperature))
    return probs


def nucleus(probs, p):
    probs /= torch.sum(probs)
    sorted_probs, sorted_index = torch.sort(probs, descending=True)
    cusum_sorted_probs = torch.cumsum(sorted_probs, dim=0)
    after_threshold = cusum_sorted_probs > p
    if torch.sum(after_threshold) > 0:
        last_index = torch.where(after_threshold)[0][0].item() + 1
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:3]  # just assign a value
    candi_probs = probs[candi_index]
    candi_probs /= torch.sum(candi_probs)
    word = torch.multinomial(candi_probs, 1).item()
    return candi_index[word]


def generate_music(model, prompt, tokens2ids, ids2tokens):
    prompt_ids = [tokens2ids[token] for token in prompt]

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(cfg.max_length), desc='Generating music'):
            inputs = torch.tensor(prompt_ids).unsqueeze(0).to(cfg.device)
            outputs = model(inputs)
            outputs = outputs.squeeze(0)
            outputs = temperature(outputs, cfg.temperature)
            predict = nucleus(outputs, cfg.nucleus)
            # predict = torch.argmax(outputs, dim=-1)
            prompt_ids.append(predict.item())

    music = [ids2tokens[str(id)] for id in prompt_ids]
    music = '\n'.join(music)

    return music


def generate():
    model = load_model(cfg.model_path)
    tokens2ids = load_file(cfg.tokens2ids_path)
    ids2tokens = load_file(cfg.ids2tokens_path)
    prompt = load_file(cfg.prompt_path).split()
    if prompt[-1] == 'end':
        prompt = prompt[:-1]
    music = generate_music(model, prompt, tokens2ids, ids2tokens)
    save_file(music, os.path.join(cfg.output_path, 'generated.txt'))


if __name__ == '__main__':
    default_config = yaml.full_load(open('generation/config.yaml', 'r'))

    parser = argparse.ArgumentParser(description='Generate music')

    parser.add_argument('--model_path', type=str, default=default_config['INFERENCE']['model_path'], help='Path to the model')
    parser.add_argument('--output_path', type=str, default=default_config['INFERENCE']['output_path'], help='Path to the output directory')
    parser.add_argument('--tokens2ids_path', type=str, default=default_config['INFERENCE']['tokens2ids_path'], help='Path to the tokens2ids file')
    parser.add_argument('--ids2tokens_path', type=str, default=default_config['INFERENCE']['ids2tokens_path'], help='Path to the ids2tokens file')
    parser.add_argument('--prompt_path', type=str, default=default_config['INFERENCE']['prompt_path'], help='Path to the prompt file')

    parser.add_argument('--max_length', type=int, default=default_config['GENERATION']['max_length'], help='Max length of the generated music')
    parser.add_argument('--temperature', type=float, default=default_config['GENERATION']['temperature'], help='Temperature for sampling')
    parser.add_argument('--nucleus', type=float, default=default_config['GENERATION']['nucleus'], help='Nucleus for sampling')

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')

    cfg = parser.parse_args()

    generate()
