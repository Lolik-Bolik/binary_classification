from utils import Scrapper, ScrapperConfig, TrainConfig, crop_face
import torch
import wandb
from torch import optim
from torch import nn
from models.model_builder import build_model
from trainer import Trainer
from losses import LabelSmoothingLoss
import numpy as np
import random


def set_seed():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(42)
    np.random.seed(42)


wandb.init(project="Keira_Natalie_classification")


def main(scrapper_opts, train_opts):
    set_seed()
    wandb.config.update(train_opts)
    # download samples from google
    if scrapper_opts.download_data:
        scrapper = Scrapper(scrapper_opts.file_with_classes, scrapper_opts.path_to_driver)
        scrapper(scrapper_opts.target_path, scrapper_opts.samples_per_class)
        crop_face(scrapper_opts.path_to_data, 'faces')

    # as the baseline, we will use squeezenet lightweight classification model
    if scrapper_opts.path_to_data:
        model = build_model(train_opts.model_name)
        criterion = LabelSmoothingLoss(2) if train_args.criterion == 'labelsmoothing' else nn.CrossEntropyLoss()
        optimizer_ft = optim.Adam(model.parameters(), lr=0.0001)
        trainer = Trainer(train_args, model, criterion, optimizer_ft, wandb)
        wandb.watch(model)
        trainer.run_training()


if __name__ == "__main__":
    train_args = TrainConfig()
    scrapper_args = ScrapperConfig(path_to_data=train_args.datapath)
    main(scrapper_args, train_args)
