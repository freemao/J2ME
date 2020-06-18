# -*- coding: UTF-8 -*-

"""
Object detection using Faster-RCNN implemented in pytorch 
"""

# code adopted from torchvision object detection tutorial 
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
import os
import sys
import time
import torch
import logging
import torchvision
import numpy as np
import pandas as pd
from PIL import Image

import J2ME.detectron.utils as utils
from J2ME.apps.base import OptionParser, ActionDispatcher, put2slurm
from J2ME.common.base import EarlyStopping 
from J2ME.common.datasets import ObjectDetectionDataset
from J2ME.detectron.engine import train_one_epoch, evaluate
import J2ME.detectron.transforms as T

from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def main():
    actions = (
        ('train', 'using pretrained model to solve regression problems'),
        ('predict', 'make predictions using the trained model'),
            )
    p = ActionDispatcher(actions)
    p.dispatch(globals())

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train(args):
    """
    %prog train train_csv, train_dir, valid_csv, valid_dir, model_name_prefix
    Args:
        train_csv: csv file containing training image 'fn' and 'targets'
        train_dir: directory where training images are located
        valid_csv: csv file containing validation image 'fn' and 'targets'
        valid_dir: directory where validation images are located
        model_name_prefix: the prefix of the output model name 
    """
    p = OptionParser(train.__doc__)
    p.add_option('--batchsize', default=2, type='int', 
                    help='batch size')
    p.add_option('--epoch', default=26, type='int', 
                    help='number of total epochs')
    p.add_option('--patience', default=20, type='int', 
                    help='patience in early stopping')
    p.add_option('--backbone', default='resnet50',
                    help='pretrained model name used as backbone')
    p.add_option('--output_dir', default='.',
                    help='directory path where to save')
    p.add_option('--disable_slurm', default=False, action="store_true",
                 help='run directly without generating slurm job')
    p.add_slurm_opts(job_prefix=train.__name__)

    opts, args = p.parse_args(args)
    if len(args) != 5:
        sys.exit(not p.print_help())
    train_csv, train_dir, valid_csv, valid_dir, model_name_prefix = args

    # genearte slurm file
    if not opts.disable_slurm:
        cmd_header = 'ml singularity'
        cmd = "singularity exec docker://unlhcc/pytorch:1.5.0 "\
            "python3 -m schnablelab.CNN.TransLearning prediction "\
            f"{saved_model} {test_csv} {test_dir} {output} "\
            f"--batchsize {opts.batchsize} --disable_slurm "
        if opts.pretrained_mn:
         cmd += f"--pretrained_mn {opts.pretrained_mn}"
        put2slurm_dict = vars(opts)
        put2slurm_dict['cmd_header'] = cmd_header
        put2slurm([cmd], put2slurm_dict)
        sys.exit()


    logfile = model_name_prefix + '.log'
    #histfile = model_name_prefix + '.hist.csv'
    logging.basicConfig(filename=logfile, level=logging.DEBUG, format="%(asctime)s:%(levelname)s:%(message)s")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.debug('device: %s'%device)
    logging.debug('pytorch version: %s'%torch.__version__)
    logging.debug('cuda version: %s'%torch.version.cuda)

    train_dataset = ObjectDetectionDataset(train_csv, train_dir, get_transform(train=True))
    valid_dataset = ObjectDetectionDataset(valid_csv, valid_dir, get_transform(train=False))
    train_loader = DataLoader(train_dataset, batch_size=opts.batchsize, collate_fn=utils.collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=opts.batchsize, collate_fn=utils.collate_fn)
    dataloaders_dict = {'train': train_loader, 'valid': valid_loader}

    model = get_model(2) # either leaf tip or background
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad] # 'requires_grad' default is Ture (will be trained)
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    
    for epoch in range(opts.epoch):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, dataloaders_dict['train'], device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()

        # save model
        utils.save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'opts': opts,
                'epoch': epoch},
                os.path.join(opts.output_dir, f'{model_name_prefix}_{epoch}.pth'))

        # evaluate on the test dataset
        evaluate(model, dataloaders_dict['valid'], device=device)

def predict(args):
    """
    %prog predict saved_model test_csv, test_dir, output
    Args:
        saved_model: saved model with either a .pt or .pth file extension
        test_csv: csv file (comma separated without header) containing all testing image filenames
        test_dir: directory where testing images are located
        output: csv file saving prediction results
    """
    p = OptionParser(predict.__doc__)
    p.add_option('--batchsize', default=2, type='int', 
                    help='batch size')
    p.add_option('--backbone', default='resnet50',
                    help='pretrained model name used as backbone')
    p.add_option('--disable_slurm', default=False, action="store_true",
                 help='run directly without generating slurm job')
    p.add_slurm_opts(job_prefix=predict.__name__)

    opts, args = p.parse_args(args)
    if len(args) != 4:
        sys.exit(not p.print_help())
    saved_model, test_csv, test_dir, output = args

    # genearte slurm file
    if not opts.disable_slurm:
        cmd_header = 'ml singularity'
        cmd = "singularity exec docker://unlhcc/pytorch:1.5.0 "\
            "python3 -m schnablelab.CNN.TransLearning prediction "\
            f"{saved_model} {test_csv} {test_dir} {output} "\
            f"--batchsize {opts.batchsize} --disable_slurm "
        if opts.pretrained_mn:
         cmd += f"--pretrained_mn {opts.pretrained_mn}"
        put2slurm_dict = vars(opts)
        put2slurm_dict['cmd_header'] = cmd_header
        put2slurm([cmd], put2slurm_dict)
        sys.exit()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = get_model(2)

    model.load_state_dict(torch.load(saved_model, map_location=device))
    model.eval()

    test_dataset = ObjectDetectionDataset(test_csv, test_dir, get_transform(train=False))
    test_loader = DataLoader(test_dataset, batch_size=opts.batchsize)

    evaluate(model, test_loader, device=device)

if __name__ == "__main__":
    main()