from utils import Scrapper, ScrapperConfig, TrainConfig, load_split_train_test, crop_face
import torch
import wandb
from torch import nn
from torch import optim
from torchvision import models
import os


wandb.init(project="Keira_Natalie_classification")


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(args, model, device, criterion, train_loader, optimizer, scheduler, epoch):
    if scheduler is not None:
        scheduler.step()
    model.train()
    train_loss = 0
    total_accuracy = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, dtype=torch.float), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss
        total_accuracy += accuracy(output, target)[0]
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    wandb.log({
        "Train Accuracy": total_accuracy / len(train_loader.dataset),
        "Train Loss": train_loss})


def test(model, criterion,  device, test_loader):
    model.eval()
    test_loss = 0
    total_accuracy = 0
    example_images = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, dtype=torch.float), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += criterion(output, target)
            total_accuracy += accuracy(output, target)[0]
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            example_images.append(wandb.Image(
                data[0], caption="Pred: {} Truth: {}".format(pred[0].item(), target[0])))
    wandb.log({
        "Examples": example_images,
        "Test Accuracy": total_accuracy / len(test_loader),
        "Test Loss": test_loss})
    return total_accuracy / len(test_loader)


def main(scrapper_opts, train_opts):
    wandb.config.update(train_opts)
    # download samples from google
    if scrapper_opts.download_data:
        scrapper = Scrapper(scrapper_opts.file_with_classes, scrapper_opts.path_to_driver)
        scrapper(scrapper_opts.target_path, scrapper_opts.samples_per_class)
        crop_face(scrapper_opts.path_to_data, 'faces')

    # as the baseline, we will use squeezenet lightweight classification model
    if scrapper_opts.path_to_data:
        trainloader, testloader = load_split_train_test(train_opts, scrapper_opts.path_to_data, .2)
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "cpu")
        model = models.squeezenet1_1(pretrained=True)
        num_of_output_classes = 2
        # change the last conv2d layer
        model.classifier._modules["1"] = nn.Conv2d(512, num_of_output_classes, kernel_size=(1, 1))
        # change the internal num_classes variable rather than redefining the forward pass
        model.num_classes = num_of_output_classes
        # num_features = model.fc.in_features
        # model.fc = nn.Linear(num_features, 2)
        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.Adam(model.parameters(), lr=0.0001)
        # exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        model.to(device)
        wandb.watch(model)
        min_accuracy = 0
        for epoch in range(1, train_opts.epochs + 1):
            train(train_opts, model, device, criterion, trainloader, optimizer_ft, None, epoch)
            test_accuracy = test(model, criterion, device, testloader)
            if test_accuracy > min_accuracy:
                torch.save(model.state_dict(), os.path.join(wandb.run.dir, "best_model.pth"))


if __name__ == "__main__":
    scrapper_args = ScrapperConfig(path_to_data='data')
    train_args = TrainConfig()
    main(scrapper_args, train_args)
