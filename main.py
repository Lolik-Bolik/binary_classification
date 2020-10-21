from utils import Scrapper, ScrapperConfig, TrainConfig, crop_face
import torch
import wandb
from torch import optim
from torch import nn
from models.model_builder import build_model
from trainer import Trainer


wandb.init(project="Keira_Natalie_classification")


def main(scrapper_opts, train_opts):
    wandb.config.update(train_opts)
    # download samples from google
    if scrapper_opts.download_data:
        scrapper = Scrapper(scrapper_opts.file_with_classes, scrapper_opts.path_to_driver)
        scrapper(scrapper_opts.target_path, scrapper_opts.samples_per_class)
        crop_face(scrapper_opts.path_to_data, 'faces')

    # as the baseline, we will use squeezenet lightweight classification model
    if scrapper_opts.path_to_data:
        model = build_model('dummy_model')
        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.Adam(model.parameters(), lr=0.0001)
        # exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        trainer = Trainer(train_args, model, criterion, optimizer_ft, wandb)
        wandb.watch(model)
        trainer.run_training()




if __name__ == "__main__":
    train_args = TrainConfig()
    scrapper_args = ScrapperConfig(path_to_data=train_args.datapath)
    main(scrapper_args, train_args)
