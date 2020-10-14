# -*- coding: UTF-8 -*-

"""
train autoencoder models
"""

import os
import sys
import time
import datetime
import logging
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torchvision
from torch import nn
import torchvision.transforms as transforms
from J2ME.common.datasets import GeneExpressionDataset
import torch.optim as optim
from torch.utils.data import DataLoader
from J2ME.AE.models import SimpleAE
from J2ME.apps.base import OptionParser, ActionDispatcher, put2slurm
from J2ME.common.base import EarlyStopping 
from schnablelab.CNN.base import EarlyStopping

def main():
    actions = (
        ('train', 'train autoencoder'),
        ('predict', 'check latent space and reconstructed results'),
            )
    p = ActionDispatcher(actions)
    p.dispatch(globals())

def train(args):
    """
    %prog train train_csv, model_name_prefix
    Args:
        train_csv: FPKM matrix csv file
            1st column is sample names, 2nd beyound is gene normalized FPKM values
        model_name_prefix: the prefix of the output model name 
    """
    p = OptionParser(train.__doc__)
    p.add_option('--latentsize', default=5, type='int',
                    help='the size of the latent feature')
    p.add_option('--batchsize', default=64, type='int', 
                    help='batch size')
    p.add_option('--epoch', default=2000, type='int', 
                    help='number of total epochs')
    p.add_option('--patience', default=20, type='int', 
                    help='patience in early stopping')
    p.add_option('--output_dir', default='.',
                    help='directory path where to save')
    p.add_option('--disable_slurm', default=False, action="store_true",
                 help='run directly without generating slurm job')
    p.add_slurm_opts(job_prefix=train.__name__)

    opts, args = p.parse_args(args)
    if len(args) != 2:
        sys.exit(not p.print_help())
    train_csv, model_name_prefix = args

    # genearte slurm file
    if not opts.disable_slurm:
        cmd = "python -m J2ME.AE.autoencoder train "\
            f"{train_csv} {model_name_prefix} --disable_slurm "
        if opts.latentsize:
            cmd += f"--latentsize {opts.latentsize} "
        if opts.batchsize:
            cmd += f"--batchsize {opts.batchsize} "
        if opts.epoch:
            cmd += f"--epoch {opts.epoch} "
        if opts.output_dir:
            cmd += f"--output_dir {opts.output_dir} "
        put2slurm_dict = vars(opts)
        put2slurm([cmd], put2slurm_dict)
        sys.exit()

    input_size = pd.read_csv(train_csv, index_col=0).shape[1]
    print('input size: %s'%input_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleAE(input_shape=input_size, output_shape=opts.latentsize).to(device)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    #criterion = nn.MSELoss()
    criterion = nn.L1Loss()

    train_dataset = GeneExpressionDataset(train_csv)
    train_loader = DataLoader(train_dataset, batch_size=opts.batchsize)

    loss_hist = []
    early_stopping = EarlyStopping(model_name_prefix, patience=opts.patience, verbose=True)
    for epoch in range(opts.epoch):
        print('Epoch {}/{}'.format(epoch, opts.epoch - 1))
        print('-' * 10)
        running_loss = 0.0
        for batch_features,_ in train_loader:
            batch_features = batch_features.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_features)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        loss = running_loss / len(train_loader)
        print('Loss: {:.4f}'.format(loss))
        early_stopping(loss, model)
        loss_hist.append(loss)
        if early_stopping.early_stop:
            print('Early stopping')
            break
        print('Best val loss: {:4f}'.format(early_stopping.val_loss_min))
    pd.DataFrame(loss_hist).to_csv('%s.loss.hist'%model_name_prefix)

def predict(args):
    """
    %prog predict saved_model predict_csv, output_prefix
    """
    p = OptionParser(predict.__doc__)
    p.add_option('--latentsize', default=5, type='int',
                    help='the size of the latent feature')
    p.add_option('--batchsize', default=64, type='int', 
                    help='batch size')
    p.add_option('--output_dir', default='.',
                    help='directory path where to save')
    p.add_option('--disable_slurm', default=False, action="store_true",
                 help='run directly without generating slurm job')
    p.add_slurm_opts(job_prefix=predict.__name__)

    opts, args = p.parse_args(args)
    if len(args) != 3:
        sys.exit(not p.print_help())
    saved_model, predict_csv, output_prefix = args
    out_dir = Path(opts.output_dir)
    if not out_dir.exists():
        out_dir.mkdir()
    input_size = pd.read_csv(train_csv, index_col=0).shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleAE(input_shape=input_size, output_shape=opts.latentsize).to(device)
    model.load_state_dict(torch.load(saved_model, map_location=device))
    model.eval()

    predict_dataset = GeneExpressionDataset(predict_csv)
    predict_loader = DataLoader(predict_dataset, batch_size=opts.batchsize)

    Original, LatentSpace, Constructed, Labels = [], [], [], []
    for batch_features, label in predict_loader:
        Original.append(batch_features)
        ls = model.encoder(batch_features).detach().numpy() 
        LatentSpace.append(ls)
        cst = model(batch_features).detach().numpy()
        Constructed.append(cst)
        Labels.append(label)
    orig = np.concatenate(Original)
    late = np.concatenate(LatentSpace)
    cons = np.concatenate(Constructed)
    labe = np.concatenate(Labels)
    
    np.save(Path(opts.output_dir)/('%s_original.npy'%output_prefix), orig)
    np.save(Path(opts.output_dir)/('%s_latent.npy'%output_prefix), late)
    np.save(Path(opts.output_dir)/('%s_constructed.npy'%output_prefix), cons)
    np.save(Path(opts.output_dir)/('%s_label.npy'%output_prefix), labe)

if __name__ == "__main__":
    main()