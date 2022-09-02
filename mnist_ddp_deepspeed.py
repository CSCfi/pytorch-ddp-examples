# Based on multiprocessing example from
# https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html

from datetime import datetime
import argparse
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import deepspeed
from datetime import timedelta


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def train(args):
    num_epochs = args.epochs
    local_rank = args.local_rank
    if local_rank == -1:
        local_rank = int(os.environ.get('PMIX_RANK', -1))

    deepspeed.init_distributed(timeout=timedelta(minutes=5))

    torch.manual_seed(0)
    model = ConvNet()

    criterion = nn.CrossEntropyLoss().cuda()

    train_dataset = MNIST(root='./data', train=True,
                          transform=transforms.ToTensor(), download=True)

    model_engine, optimizer, train_loader, __ = deepspeed.initialize(
        args=args, model=model, model_parameters=model.parameters(),
        training_data=train_dataset)

    start = datetime.now()
    for epoch in range(num_epochs):
        tot_loss = 0
        for i, data in enumerate(train_loader):
            images = data[0].to(model_engine.local_rank)
            labels = data[1].to(model_engine.local_rank)

            outputs = model_engine(images)
            loss = criterion(outputs, labels)

            model_engine.backward(loss)
            model_engine.step()

            tot_loss += loss.item()

        if local_rank == 0:
            print('Epoch [{}/{}], average loss: {:.4f}'.format(
                epoch + 1,
                num_epochs,
                tot_loss / (i+1)))

    if local_rank == 0:
        print("Training completed in: " + str(datetime.now() - start))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')

    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
