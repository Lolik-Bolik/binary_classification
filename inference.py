import argparse
from models.model_builder import build_model
from utils import crop_face
import torch
import cv2
import logging as logger
import os
from utils import draw_classification_legend
from time import time

logger.basicConfig(level=logger.INFO)
classes = ['Keira Knightley', 'Natalie Portman']


def run(opts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('Building the model ...')
    model = build_model(opts.model_name)
    model.load_state_dict(torch.load(opts.model_path))
    model.to(device).eval()
    logger.info('Model was successfully built and loaded !')
    logger.info('Read input image ...')
    input_image = cv2.imread(opts.input_image)
    image_name = os.path.basename(opts.input_image)
    print(image_name)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    face, prob = crop_face(img=input_image)
    total_time = 0
    if prob >= opts.face_threshold:
        if opts.fps_count:
            for i in range(110):
                # warmup
                if i < 10:
                    pass
                start_time = time()
                input_tensor = face.to(device, dtype=torch.float).unsqueeze(0)
                output = model(input_tensor)
                total_time += round(time() - start_time, 4)
            fps = round(100 / total_time, 2)
            logger.info(fps)

        input_tensor = face.to(device, dtype=torch.float).unsqueeze(0)
        output = model(input_tensor)

        results = {cls: round(score.item(), 3) for cls, score in zip(classes, output.softmax(dim=1).squeeze(0))}
        image = draw_classification_legend(image=input_image, class_map=results)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'results/{image_name}', image)
        # cv2.imshow('result', image)
        # cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify input image')
    parser.add_argument('--input_image', type=str, help='path to input image')
    parser.add_argument('--model_path', type=str, help='path to model')
    parser.add_argument('--model_name', type=str, help='model name'
                                                       'could be "squeezenet" or "dummy_model"')
    parser.add_argument('--face_threshold', type=str, default=0.7, help='threshold')
    parser.add_argument('--fps_count', action='store_true')
    args = parser.parse_args()
    run(args)