import torch
from utils import load_split_train_test, accuracy, Logger, AverageMeter
import os


class Trainer:
    def __init__(self, args, model, criterion, optimizer, wandb, scheduler=None):
        self.args = args
        self.model = model
        self.criterion = criterion
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available()
                              else "cpu")
        self.train_loader, self.test_loader = load_split_train_test(args)
        self.loss = {'train': AverageMeter(), 'test': AverageMeter()}
        self.accuracy = {'train': AverageMeter(), 'test': AverageMeter()}
        self.logger = Logger(wandb, args, len(self.train_loader.dataset))
        self.epoch = 0
        self.min_accuracy = 0

    def before_training_step(self):
        self.model.train()
        self.epoch += 1

        if self.scheduler is not None:
            self.scheduler.step()

    def after_training_step(self):
        self.logger.epoch_log(self.accuracy, self.loss, 'train')

    def training_step(self):
        self.before_training_step()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device, dtype=torch.float), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            self.loss['train'].update(loss.item())
            self.accuracy['train'].update(accuracy(output, target)[0])
            loss.backward()
            self.optimizer.step()
            self.logger.batch_log(self.epoch, batch_idx, loss.item())

        self.after_training_step()

    def before_validation_step(self):
        self.model.eval()

    def after_validation_step(self):
        if self.accuracy['test'].avg > self.min_accuracy:
            torch.save(self.model.state_dict(), os.path.join(self.logger.wandb.run.dir, "best_model.pth"))
            self.min_accuracy = self.accuracy['test'].avg
        self.logger.epoch_log(self.accuracy, self.loss, 'test')
        self.loss = {stage: meter.reset() for stage, meter in self.loss.items()}
        self.accuracy = {stage: meter.reset() for stage, meter in self.accuracy.items()}

    def validation_step(self):
        self.before_validation_step()
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device, dtype=torch.float), target.to(self.device)
                output = self.model(data)
                self.loss['test'].update(self.criterion(output, target).item())
                self.accuracy['test'].update(accuracy(output, target)[0])
        self.after_validation_step()

    def run_training(self):
        self.model.to(self.device)
        for _ in range(1, self.args.epochs + 1):
            self.training_step()
            self.validation_step()
