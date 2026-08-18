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

datadir = os.path.join('/scratch', os.environ['SLURM_JOB_ACCOUNT'],
                       os.environ['USER'], 'MNIST')


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
    deepspeed.init_distributed(timeout=timedelta(minutes=5))

    model = ConvNet()

    criterion = nn.CrossEntropyLoss()

    train_dataset = MNIST(root=datadir, train=True,
                          transform=transforms.ToTensor(), download=True)

    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        args=args, model=model, model_parameters=model.parameters(),
        training_data=train_dataset)

    start = datetime.now()
    for epoch in range(args.epochs):
        tot_loss = 0
        for i, data in enumerate(train_loader):
            images = data[0].to(model_engine.local_rank)
            labels = data[1].to(model_engine.local_rank)

            outputs = model_engine(images)
            loss = criterion(outputs, labels)

            model_engine.backward(loss)
            model_engine.step()

            tot_loss += loss.item()

        # if model_engine.rank == 0:
        #     print('Epoch [{}/{}], average loss: {:.4f}'.format(
        #         epoch + 1,
        #         args.epochs,
        #         tot_loss / (i+1)))
    if model_engine.global_rank == 0:
        print("Training completed in: " + str(datetime.now() - start))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=2, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')    
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loader workers')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
