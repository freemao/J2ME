# -*- coding: UTF-8 -*-

"""
Prepare customized datasets for training pytorch models
"""
import sys
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

class GeneExpressionDataset(Dataset):
    """
    prepare gene expression dataset.
    """
    def __init__(self, csv_fn, transform=None):
        """
        Args:
            csv_fn (string): FPKM csv matrix file with dimension (# of indi, # of genes+1)
                tab separated
                first col: sample names
                2nd beyond: gene names
        """
        self.csv_df = pd.read_csv(csv_fn, index=0) # the sample column will be index
        self.transform = transform

    def __len__(self):
        return self.csv_df.shape[0] # number of individuals

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        fpkms = self.csv_df.iloc[:, idx].astype(np.float32).values
        label = self.csv_df.columns[idx]
        return torch.tensor(fpkms), label

class LeafcountingDataset(Dataset):
    """
    prepare leaf counting dataset.
    """
    def __init__(self, csv_fn, root_dir, transform=None):
        """
        Args:
            csv_fn (string): 
                Path to the comma separated csv file with header. 
                1st column ('fn') is image file name
                2nd column ('label) is the annotation/label. 
            root_dir (string): Directory where all the images in csv file lie.
        """
        self.csv_df = pd.read_csv(csv_fn)
        if 'fn' not in self.csv_df.columns or 'label' not in self.csv_df.columns:
            sys.exit("Couldn't find 'fn' and 'label' in the csv header.")
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.csv_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_fn = self.csv_df.loc[idx, 'fn']
        img = Image.open(self.root_dir/img_fn)
        if len(img.getbands()) == 4:
            img = img.convert('RGB')
        label = self.csv_df.loc[idx, 'label'].astype('float32').reshape(-1,)

        if self.transform:
            img = self.transform(img)

        return img, label, img_fn

    def __init__(self, csv_fn, root_dir, transform=None):
        """
        Args:
            csv_fn (string): 
                Path to the comma separated csv file with header. 
                1st column ('fn') is image file name
                2nd column ('label) is the annotation/label. 
            root_dir (string): Directory where all the images in csv file lie.
        """
        self.csv_df = pd.read_csv(csv_fn)
        if 'fn' not in self.csv_df.columns or 'label' not in self.csv_df.columns:
            sys.exit("Couldn't find 'fn' and 'label' in the csv header.")
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.csv_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_fn = self.csv_df.loc[idx, 'fn']
        img = Image.open(self.root_dir/img_fn)
        if len(img.getbands()) == 4:
            img = img.convert('RGB')
        label = self.csv_df.loc[idx, 'label'].astype('float32').reshape(-1,)

        if self.transform:
            img = self.transform(img)

        return img, label, img_fn

class ObjectDetectionDataset(Dataset):
    def __init__(self, csv_fn, root_dir, transforms, sep=',', only_image=False):
        '''
        csv_fn: tab separated csv file with:
            1st column('fn'): image file name
            2st column('targets): target information including x, y, and label in json format
        root_dir:
            where the images in csv file are located
        '''
        self.csv_df = pd.read_csv(csv_fn, sep=sep)
        if only_image:
            if 'fn' not in self.csv_df.columns:
                sys.exit("Couldn't find 'fn' in the csv header.")
        else:
            if 'fn' not in self.csv_df.columns or 'targets' not in self.csv_df.columns:
                sys.exit("Couldn't find 'fn' and 'targets' in the csv header.")

        self.root_dir = Path(root_dir)
        self.transforms = transforms
        self.only_image = only_image

    def __len__(self):
        return len(self.csv_df)

    def __getitem__(self, idx):
        #print(f'idx: {idx}')
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_fn = self.csv_df.loc[idx, 'fn']
        #print(f'img_fn: {img_fn}')
        img = Image.open(self.root_dir/img_fn)
        target = {}
        if len(img.getbands()) == 4:
            img = img.convert('RGB')
        
        if self.only_image:
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            return img, target, img_fn

        # map each label to a class with name started from 1
        tips = json.loads(self.csv_df.loc[idx, 'targets'])
        #print(tips)
        #label_series = pd.Series(np.unique(tips['label']))
        #label_series.index += 1
        #label_dict = {y:x for x,y in label_series.items()}
        #print(label_dict)
        label_dict = {'intact_tip':1, 'cut_tip':2}

        boxes, labels = [], []
        for x, y, label in zip(tips['x'], tips['y'], tips['label']):
            boxes.append([x-15, y-15, x+15, y+15]) # square with side length==30
            labels.append(label_dict[label])
        boxes = torch.tensor(boxes, dtype=torch.float)
        labels = torch.tensor(labels)
        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        image_id = torch.tensor([idx])
        # zeros: False, ones: True (will not be used in evaluation)
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        #print(boxes.size(), labels.size(), image_id.size(), areas.size(), iscrowd.size())

        
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = areas
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        #print(img.size(), len(target))

        return img, target , img_fn

class InstanceSegmentationDataset(Dataset):
    '''
    add mask information
    '''
    pass

class KeypointDataset(Dataset):
    '''
    add key points information
    '''
    pass
