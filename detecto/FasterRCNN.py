# -*- coding: UTF-8 -*-

"""
Object detection using Faster-RCNN implemented in pytorch 
"""

# code adopted from torchvision object detection tutorial 
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from J2ME.apps.base import OptionParser, ActionDispatcher, put2slurm
from J2ME.common.base import EarlyStopping 
from J2ME.common.datasets import ObjectDetectionDataset
from J2ME.common.draw import show_box
from J2ME.detecto.engine import train_one_epoch, evaluate
import J2ME.detecto.utils as utils
import J2ME.detecto.transforms as T

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

def get_model(num_classes, backbone='resnet50'):
    if backbone == 'resnet50':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    elif backbone == 'resnet101':
        from J2ME.detecto.models import fasterrcnn_resnet101_fpn
        model = fasterrcnn_resnet101_fpn(pretrained=True)

    else:
        sys.exit('only resnet50 and resnet101 supported!')

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train(args):
    """
    %prog train train_csv, train_dir, valid_csv, valid_dir, mn_prefix
    Args:
        train_csv: csv file containing training image 'fn' and 'targets'
        train_dir: directory where training images are located
        valid_csv: csv file containing validation image 'fn' and 'targets'
        valid_dir: directory where validation images are located
        mn_prefix: the prefix of the output model name 
    """
    p = OptionParser(train.__doc__)
    p.add_option('--batchsize', default=20, type='int', 
                    help='batch size')
    p.add_option('--epoch', default=200, type='int', 
                    help='number of total epochs')
    p.add_option('--patience', default=10, type='int', 
                    help='patience in early stopping')
    p.add_option('--backbone', default='resnet50', choices=('resnet50', 'resnet101'),
                    help='pretrained model name used as backbone')
    p.add_option('--disable_slurm', default=False, action="store_true",
                 help='run directly without generating slurm job')
    p.add_slurm_opts(job_prefix=train.__name__)

    opts, args = p.parse_args(args)
    if len(args) != 5:
        sys.exit(not p.print_help())
    train_csv, train_dir, valid_csv, valid_dir, mn_prefix = args

    # genearte slurm file
    if not opts.disable_slurm:
        cmd = "python -m J2ME.detecto.fasterRCNN train "\
            f"{train_csv} {train_dir} {valid_csv} {valid_dir} {mn_prefix} "\
            f"--batchsize {opts.batchsize} "\
            f"--epoch {opts.epoch} "\
            f"--patience {opts.patience} "\
            f"--backbone {opts.backbone} --disable_slurm "
        put2slurm_dict = vars(opts)
        put2slurm([cmd], put2slurm_dict)
        sys.exit()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device: %s\npytorch version: %s\ncuda version: %s'%(device, torch.__version__, torch.version.cuda))

    train_dataset = ObjectDetectionDataset(train_csv, train_dir, get_transform(train=True))
    valid_dataset = ObjectDetectionDataset(valid_csv, valid_dir, get_transform(train=False))
    train_loader = DataLoader(train_dataset, batch_size=opts.batchsize, collate_fn=utils.collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=opts.batchsize, collate_fn=utils.collate_fn)
    dataloaders_dict = {'train': train_loader, 'valid': valid_loader}

    model = get_model(num_classes=3, backbone=opts.backbone) # healthy leaf tip, cut leaf tip, background
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad] # 'requires_grad' default is Ture (will be trained)
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    print('start training...')
    start_time = time.time()

    early_stopping = EarlyStopping(mn_prefix, patience=opts.patience, verbose=True, delta=0.01)
    for epoch in range(opts.epoch):
        loss_dict, total_loss = train_one_epoch(model, optimizer, dataloaders_dict['train'], device, epoch, print_freq=1)
        print('sum of losses: %s'%total_loss)
        print('loss dict:\n', loss_dict)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        coco_evaluator = evaluate(model, dataloaders_dict['valid'], device=device)
        ap = coco_evaluator.coco_eval['bbox'].stats[0]
        #ious = coco_evaluator.coco_eval['bbox'].ious
        print('average AP: %s'%ap)
        valid_loss = 1-ap
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print('Early stopping')
            break
        print('Best average precision: {:0.3f}'.format(1-early_stopping.val_loss_min))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def predict(args):
    """
    %prog predict saved_model test_csv, test_dir, output_prefix
    Args:
        saved_model: saved model with either a .pt or .pth file extension
        test_csv: csv file (comma separated without header) containing all testing image filenames
        test_dir: directory where testing images are located
        output: csv file saving prediction results
    """
    p = OptionParser(predict.__doc__)
    p.add_option('--backbone', default='resnet50', choices=('resnet50','resnet101'),
                    help='pretrained model name used as backbone')
    p.add_option('--score_cutoff', type='float', default=0.7,
                    help='set score cutoff')
    p.add_option('--show_box', default=False, action="store_true",
                 help = 'generate image with predicted bounding box.')
    p.add_option('--output_dir', default='.',
                 help = 'specify the output directory for saving prediction results.')
    p.add_option('--disable_slurm', default=False, action="store_true",
                 help='run directly without generating slurm job')
    p.add_slurm_opts(job_prefix=predict.__name__)

    opts, args = p.parse_args(args)
    if len(args) != 4:
        sys.exit(not p.print_help())
    saved_model, test_csv, test_dir, output_prefix = args

    out_dir = Path(opts.output_dir)
    if not out_dir.exists():
        print('output dir %s does not exist, created.'%out_dir)
        out_dir.mkdir()

    if not opts.disable_slurm:
        cmd = "python -m J2ME.detecto.fasterRCNN predict "\
            f"{saved_model} {test_csv} {test_dir} {output_prefix} "\
            f"--backbone {opts.backbone} "\
            f"--score_cutoff {opts.score_cutoff} "\
            f"--output_dir {opts.output_dir} "\
            f"--show_box {opts.show_box} --disable_slurm "
        put2slurm_dict = vars(opts)
        put2slurm([cmd], put2slurm_dict)
        sys.exit()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device detected: {device}')
    model = get_model(num_classes=3, backbone=opts.backbone)
    model.load_state_dict(torch.load(saved_model, map_location=device))
    model.eval()

    test_dataset = ObjectDetectionDataset(test_csv, test_dir, get_transform(train=False), only_image=True, sep=',')
    test_loader = DataLoader(test_dataset, batch_size=1)

    filenames, boxes, labels, scores, lcs = [], [], [], [], [] # lcs: leaf counts
    print('start testing...')
    for imgs, _, fns in test_loader:
        imgs = imgs.to(device)
        results = model(imgs)
        fn = fns[0]
        print(fn)
        boxes, labels, scores = results[0]['boxes'], results[0]['labels'], results[0]['scores']
        boxes = np.array([i.to(device).tolist() for i in boxes])
        labels = np.array([i.to(device).tolist() for i in labels])
        scores = np.array([i.to(device).tolist() for i in scores])
        idxs = np.argwhere(scores>opts.score_cutoff).squeeze()

        if opts.show_box:
            img = show_box(Path(test_dir)/fn, boxes[idxs], labels[idxs], scores[idxs])
            img_out_fn = fn.replace('.png', '.prd.jpg') if fn.endswith('.png') else fn.replace('.jpg', '.prd.jpg')
            img.save(out_dir/img_out_fn)

        filenames.append(fn)
        lcs.append(len(idxs))
    pd.DataFrame(dict(zip(['fn', 'lc'], [filenames, lcs]))).to_csv(out_dir/('%s.prediction.csv'%output_prefix), index=False)
    print('Done! check leaf counting results in %s.prediction.csv'%output_prefix)

if __name__ == "__main__":
    main()
