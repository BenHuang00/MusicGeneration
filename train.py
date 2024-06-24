import os
import argparse

import datetime

import torch
from torch import nn
from torch.utils.data import random_split
from torch.utils.data import DataLoader

import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt

import config
from utils import load_file, save_file, make_file_path
from models import *


def draw_loss_curve(history, model_name):
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(f'{cfg.train_log_path}', f'{cfg.loss_plot_name_template.format(model_name)}'))


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
                   os.path.join(f'{cfg.trained_model_path}',
                                f'{cfg.trained_model_paras_name_template.format(model.__class__.__name__, epoch + 1)}')
                   )
        save_file(history,
                  os.path.join(f'{cfg.train_history_path}',
                               f'{cfg.train_history_name_template.format(model.__class__.__name__, epoch + 1)}')
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
    wandb.login(key="")
    wandb.init(project='MIR-Project-Test')

    dataset = load_file(os.path.join(cfg.preprocess_path, cfg.preprocess_dataset_name))
    print(f'Loaded dataset: {len(dataset)} windows')

    train_size = int(len(dataset) * cfg.train_size_scale)
    val_size = int(len(dataset) * cfg.val_size_scale)
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    num_tokens = dataset.num_tokens

    os.makedirs(cfg.trained_model_path, exist_ok=True)  # TODO: make_file_path
    model = LSTM_Model(num_tokens, cfg.embedding_dim, cfg.hidden_dim, cfg.num_layers).to(cfg.device)
    train_model(model, train_loader, val_loader)
    test_model(model, test_loader)

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')

    parser.add_argument('--preprocess_path', type=str, default=config.preprocess_path,
                        help='Path to the preprocessed dataset')

    args = parser.parse_args()

    cfg = config

    cfg.preprocess_path = args.preprocess_path

    train()
