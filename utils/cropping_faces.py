from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import os
from os import walk
import cv2
from tqdm import tqdm


def crop_face(datadir, path_to_save):
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(image_size=224, device=device)
    for dirpath, dirnames, filenames in tqdm(walk(datadir)):
        if dirnames:
            for dirname in dirnames:
                if not os.path.exists(os.path.join(path_to_save, dirname)):
                    os.makedirs(os.path.join(path_to_save, dirname))
        if filenames:
            for file in filenames:
                class_type = dirpath.split('/')[-1]
                image = cv2.imread(os.path.join(dirpath, file))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                save_path = os.path.join(path_to_save, class_type, file)
                mtcnn(image, save_path=save_path)
