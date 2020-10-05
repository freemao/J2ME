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
import logging
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

def get_model(num_classes=3):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
'''    
def get_model(num_classes, backbone='resnet50',
                s=((30),) , ar=((1.0),),
                box_nms_thresh = 0.3):
    anchor_generator = AnchorGenerator(sizes=s, aspect_ratios=ar)

    if backbone == 'resnet50':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True, 
                                                        rpn_anchor_generator=anchor_generator,
                                                        box_nms_thresh = box_nms_thresh)
    elif backbone == 'resnet101':
        from J2ME.detecto.models import fasterrcnn_resnet101_fpn
        model = fasterrcnn_resnet101_fpn(pretrained=True, progress=True,
                                        rpn_anchor_generator=anchor_generator,
                                        box_nms_thresh = box_nms_thresh)
    else:
        sys.exit('only resnet50 and resnet101 supported!')

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
'''
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

    model = get_model(3) # either leaf tip or background
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad] # 'requires_grad' default is Ture (will be trained)
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    print('start training')
    start_time = time.time()
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
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

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
    p.add_option('--backbone', default='resnet50',
                    help='pretrained model name used as backbone')
    p.add_option('--score_cutoff', type='float', default=0.9,
                    help='set score cutoff')
    p.add_option('--disable_slurm', default=False, action="store_true",
                 help='run directly without generating slurm job')
    p.add_option('--show_box', default=False, action="store_true",
                 help = 'generate image with bounding box.')
    p.add_slurm_opts(job_prefix=predict.__name__)

    opts, args = p.parse_args(args)
    if len(args) != 4:
        sys.exit(not p.print_help())
    saved_model, test_csv, test_dir, output = args

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
    print(f'device detected: {device}')
    model = get_model(num_classes=3)
    checkpoint = torch.load(saved_model, map_location={'cuda:0': 'cpu'})

    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()

    test_dataset = ObjectDetectionDataset(test_csv, test_dir, get_transform(train=False), only_image=True, sep=',')
    test_loader = DataLoader(test_dataset, batch_size=1)

    filenames, boxes, labels, scores, lcs = [], [], [], [], []
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
            img.save(img_out_fn)

        filenames.append(fn)
        lcs.append(len(idxs))
    pd.DataFrame(dict(zip(['fn', 'lc'], [filenames, lcs]))).to_csv('%s.prediction.csv'%output, index=False)


if __name__ == "__main__":
    main()
