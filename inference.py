import argparse
from models.model_builder import build_model
from utils import crop_face
import torch
import numpy as np
import cv2
import logging as logger
import albumentations as A
from albumentations.pytorch import ToTensorV2
logger.basicConfig(level=logger.INFO)

test_transforms = A.Compose([
    A.Resize(227, 227),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),

])

classes = \
    {
        0: 'Keira Knightley',
        1: 'Natalie Portman'
    }


def run(opts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('Building the model ...')
    model = build_model(opts.model_name)
    model.load_state_dict(torch.load(opts.model_path))
    model.to(device).eval()
    logger.info('Model was successfully built and loaded !')
    logger.info('Read input image ...')
    input_image = cv2.imread(opts.input_image)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    face, prob = crop_face(img=input_image)
    if prob >= opts.face_threshold:
        augmented_face = test_transforms(image=np.array(face))
        input_tensor = augmented_face['image']
        input_tensor = input_tensor.to(device, dtype=torch.float).squeeze(0)
        output = model(input_tensor)
        pred = output.max(1, keepdim=True)[1]
        print(pred)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify input image')
    parser.add_argument('--input_image', type=str, help='path to input image')
    parser.add_argument('--model_path', type=str, help='path to model')
    parser.add_argument('--model_name', type=str, help='model name'
                                                       'could be "squeezenet" or "dummy_model"')
    parser.add_argument('--face_threshold', type=str,default=0.7, help='threshold')


    args = parser.parse_args()
    run(args)