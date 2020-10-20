from utils import Scrapper, Config, load_split_train_test, crop_face
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2

def main(opts):
    # download samples from google
    if opts.download_data:
        scrapper = Scrapper(opts.file_with_classes, opts.path_to_driver)
        scrapper(opts.target_path, opts.samples_per_class)
        crop_face(opts.path_to_data, 'faces')

    # simple resnet50
    if opts.path_to_data:
        trainloader, testloader = load_split_train_test(opts.path_to_data, .2)
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "cpu")
        model = models.resnet50(pretrained=True)
        model.to(device)





if __name__ == "__main__":
    args = Config(path_to_data='images')
    main(args)
