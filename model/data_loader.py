import random
import os

import numpy as np
from PIL import Image
import scipy.io as sio
import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader
import torchvision.transforms as transforms

# Define a training image loader that specifies transforms on images
train_transformer = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize the image to 224x224
    transforms.ToTensor(),  # Transform it into a torch tensor
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalize
])

mask_transformer = transforms.Compose([
    transforms.Resize((512, 512))
])

# Define a evaluation image loader that specifies transforms on images
eval_transformer = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize the image to 224x224
    transforms.ToTensor(),  # Transform it into a torch tensor
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalize
])

class GetSBDTrainDataset(Dataset):
    def __init__(self, num_classes, transform, mask_transform, data_dir="../../Dataset/SBD/dataset/img", mask_dir="../../Dataset/SBD/dataset/cls", dataset_type="../../Dataset/SBD/dataset/train_aug.txt"):
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.dataset_type = dataset_type
        self.dataset = []
        self.num_classes = num_classes
        with open(self.dataset_type, 'r') as f:
            for line in f.readlines():
                self.dataset.append('{}.mat'.format(line.strip()))
        self.mask_filenames = os.listdir(mask_dir)
        self.mask_filenames = [os.path.join(mask_dir, f) for f in self.dataset]
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.mask_filenames)

    def __getitem__(self, idx):
        mask = sio.loadmat(self.mask_filenames[idx])
        mask = Image.fromarray(mask['GTcls'][0]['Segmentation'][0])
        image = Image.open(self.mask_filenames[idx].replace('mat', 'jpg').replace(self.mask_dir, self.data_dir))
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        mask = np.array(mask)
        mask = np.where(mask==255,0,mask)
        mask = torch.from_numpy(mask)
        return image, mask

class GetCOCOTrainDataset(Dataset):
    def __init__(self, num_classes, transform, mask_transform, data_dir="../../Dataset/COCO/train2017/train2017", mask_dir=""):
        pass

class GetVOCTrainDataset(Dataset):
    def __init__(self, data_dir, mask_dir, dataset_type, num_classes, transform, mask_transform):
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.dataset_type = dataset_type
        self.dataset = []
        self.num_classes = num_classes
        with open(self.dataset_type, 'r') as f:
            for line in f.readlines():
                self.dataset.append('{}.png'.format(line.strip()))
        self.mask_filenames = os.listdir(mask_dir)
        self.mask_filenames = [os.path.join(mask_dir, f) for f in self.dataset]
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.mask_filenames)

    def __getitem__(self, idx):
        mask = Image.open(self.mask_filenames[idx])
        image = Image.open(self.mask_filenames[idx].replace('png', 'jpg').replace(self.mask_dir, self.data_dir))
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        mask = np.array(mask)
        mask = np.where(mask==255,0,mask)
        mask = torch.from_numpy(mask)
        return image, mask

class GetVOCTestDataset(Dataset):
    def __init__(self, data_dir, mask_dir, dataset_type, num_classes, transform, mask_transform):
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.dataset_type = dataset_type
        self.dataset = []
        self.num_classes = num_classes
        with open(self.dataset_type, 'r') as f:
            for line in f.readlines():
                self.dataset.append('{}.png'.format(line.strip()))
        self.mask_filenames = os.listdir(mask_dir)
        self.mask_filenames = [os.path.join(mask_dir, f) for f in self.dataset]
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.mask_filenames)

    def __getitem__(self, idx):
        mask = Image.open(self.mask_filenames[idx])
        image = Image.open(self.mask_filenames[idx].replace('png', 'jpg').replace(self.mask_dir, self.data_dir))
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        mask = np.array(mask)
        mask = np.where(mask==255,0,mask)
        mask = torch.from_numpy(mask)
        return image, mask

def fetch_dataloader(types, data_dir, mask_dir, dataset_dir, num_classes, params):
    dataloaders={}

    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(data_dir)
            mask_path = os.path.join(mask_dir)
            dataset_type = os.path.join(dataset_dir, "{}.txt".format(split))

            if split == 'train':
                VOC = GetVOCTrainDataset(path, mask_path, dataset_type, num_classes, eval_transformer, mask_transformer)
                SBD = GetSBDTrainDataset(num_classes, train_transformer, mask_transformer)
                dl = DataLoader(ConcatDataset([VOC, SBD]), batch_size=params.batch_size, shuffle=True,
                                            num_workers=params.num_workers,
                                            pin_memory=params.cuda)
            else:
                dl = DataLoader(GetVOCTestDataset(path, mask_path, dataset_type, num_classes, eval_transformer, mask_transformer), batch_size=params.batch_size, shuffle=False,
                                            num_workers=params.num_workers,
                                            pin_memory=params.cuda)
            dataloaders[split] = dl
    return dataloaders