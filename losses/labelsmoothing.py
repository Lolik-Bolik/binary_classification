import torch.nn as nn
import torch

class LabelSmoothingLoss(nn.Module):
    def __init__(self, n_classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = n_classes
        self.dim = dim

    def forward(self, output, target, *args):
        output = output.log_softmax(dim=self.dim)
        with torch.no_grad():
            # Create matrix with shapes batch_size x n_classes
            true_dist = torch.zeros_like(output)
            # Initialize all elements with epsilon / N - 1
            true_dist.fill_(self.smoothing / (self.cls - 1))
            # Fill correct class for each sample in the batch with 1 - epsilon
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * output, dim=self.dim))