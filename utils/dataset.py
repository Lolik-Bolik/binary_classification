import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import albumentations as A
from torch.utils.data import Dataset
import cv2
from albumentations.pytorch import ToTensorV2
np.random.seed(42)


class AlbumentationsDataset(datasets.ImageFolder):

    def __init__(self, root, transform=None):
        super(AlbumentationsDataset, self).__init__(root, transform)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)
            image = image['image']
        return image, target

    def __len__(self):
        return len(self.samples)


def load_split_train_test(train_opts, valid_size=.2):
    datadir = train_opts.datapath
    train_transforms = A.Compose([
        A.Resize(227, 227),
        A.ColorJitter(p=0.5),
        A.OneOf([
            A.GaussianBlur(p=1),
            A.GaussNoise(p=1),
        ], p=0.8),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),

    ])
    test_transforms = A.Compose([
        A.Resize(227, 227),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),

    ])

    train_data = AlbumentationsDataset(datadir, transform=train_transforms)
    test_data = AlbumentationsDataset(datadir, transform=test_transforms)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                                              sampler=train_sampler, batch_size=train_opts.train_batch_size, num_workers=train_opts.num_workers)
    testloader = torch.utils.data.DataLoader(test_data,
                                             sampler=test_sampler, batch_size=train_opts.test_batch_size, num_workers=train_opts.num_workers)
    return trainloader, testloader