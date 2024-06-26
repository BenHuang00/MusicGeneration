import os
import sys
import argparse
import yaml

import datetime

import torch
from torch import nn
from torch.utils.data import random_split
from torch.utils.data import DataLoader

import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.utils import load_file, save_file
from models import *


def draw_loss_curve(history, model_name):
    plt.plot([i + 1 for i in range(len(history['train_loss']))], history['train_loss'], label='Train Loss')
    plt.plot([i + 1 for i in range(len(history['val_loss']))], history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(f'{cfg.output_path}/figures', f'{model_name}_loss_curve.png'))


def train_model(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    wandb.watch(model, criterion, log="all", log_freq=10)

    history = {'train_loss': [], 'val_loss': []}

    print(f'Start training {model.__class__.__name__} model')

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0
        for i, (inputs, targets) in tqdm(enumerate(train_loader), desc=f'Epoch {epoch + 1}/{cfg.epochs}', total=len(train_loader)):
            inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            wandb.log({"train_loss": loss.item(), 'step': i + 1 + epoch * len(train_loader)})
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (inputs, targets) in tqdm(enumerate(val_loader), desc=f'Validation', total=len(val_loader)):
                inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        wandb.log({"train_loss": train_loss, "val_loss": val_loss, 'epoch': epoch + 1})
        print(
            f'Epoch {epoch + 1}/{cfg.epochs}({datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}) - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')

        torch.save(model.state_dict(),
                   os.path.join(f'{cfg.output_path}/models',
                                f'{model.__class__.__name__}_model-{epoch + 1}.pth')
                   )
        save_file(history,
                  os.path.join(f'{cfg.output_path}/history',
                               f'{model.__class__.__name__}_history-{epoch + 1}.json')
                  )

    draw_loss_curve(history, model.__class__.__name__)

    print('Training completed')


def test_model(model, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    with torch.no_grad():
        for i, (inputs, targets) in tqdm(enumerate(test_loader), desc=f'Testing', total=len(test_loader)):
            inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
        test_loss /= len(test_loader)
        print(f'Test Loss: {test_loss:.4f}')


def train():
    wandb.login()
    wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity)

    dataset = load_file(os.path.join(cfg.preprocess_path, 'gpdataset.pkl'))
    print(f'Loaded dataset: {len(dataset)} windows')

    train_size = int(len(dataset) * cfg.train_scale)
    val_size = int(len(dataset) * cfg.val_scale)
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    num_tokens = dataset.num_tokens

    os.makedirs(cfg.output_path, exist_ok=True)  # TODO: make_file_path
    os.makedirs(os.path.join(cfg.output_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_path, 'history'), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_path, 'figures'), exist_ok=True)

    model_config[cfg.model]['num_tokens'] = num_tokens
    model = eval(cfg.model)(model_config[cfg.model]).to(cfg.device)

    train_model(model, train_loader, val_loader)
    test_model(model, test_loader)

    wandb.finish()


def check_config():
    assert cfg.model in model_config, f'Unsupported model: {cfg.model}'
    assert os.path.exists(cfg.preprocess_path), f'Not found: {cfg.preprocess_path}'
    assert 0 < cfg.train_scale < 1, f'Invalid train_scale: {cfg.train_scale}'
    assert 0 < cfg.val_scale < 1, f'Invalid val_scale: {cfg.val_scale}'
    assert cfg.train_scale + cfg.val_scale < 1, f'Invalid train_scale and val_scale: {cfg.train_scale}, {cfg.val_scale}'
    assert cfg.batch_size > 0, f'Invalid batch_size: {cfg.batch_size}'
    assert cfg.epochs > 0, f'Invalid epochs: {cfg.epochs}'
    assert cfg.lr > 0, f'Invalid lr: {cfg.lr}'
    print('Config checked')


if __name__ == '__main__':
    default_config = yaml.full_load(open('train/config.yaml', 'r'))

    parser = argparse.ArgumentParser(description='Train model')

    parser.add_argument('--preprocess_path', type=str, default=default_config['INFERENCE']['preprocess_path'], help='Path to the preprocessed dataset')
    parser.add_argument('--model_config_path', type=str, default=default_config['INFERENCE']['model_config_path'], help='Path to the model config')
    parser.add_argument('--output_path', type=str, default=default_config['INFERENCE']['output_path'], help='Path to the output directory')

    parser.add_argument('--model', type=str, default=default_config['TRAIN']['model'], help='Model name')

    parser.add_argument('--train_scale', type=float, default=default_config['TRAIN']['train_scale'], help='Scale of the training set')
    parser.add_argument('--val_scale', type=float, default=default_config['TRAIN']['val_scale'], help='Scale of the validation set')
    parser.add_argument('--batch_size', type=int, default=default_config['TRAIN']['batch_size'], help='Batch size')

    parser.add_argument('--epochs', type=int, default=default_config['TRAIN']['epochs'], help='Number of epochs')
    parser.add_argument('--lr', type=float, default=default_config['TRAIN']['lr'], help='Learning rate')

    wandb_config = yaml.full_load(open('wandb_config.yaml', 'r'))

    parser.add_argument('--wandb_project', type=str, default=wandb_config['WANDB']['wandb_project'], help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=wandb_config['WANDB']['wandb_entity'], help='Wandb entity name')
    parser.add_argument('--wandb_key', type=str, default=wandb_config['WANDB']['wandb_key'], help='Wandb key')

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')

    cfg = parser.parse_args()

    model_config = yaml.full_load(open('models/config.yaml', 'r'))

    check_config()

    train()
