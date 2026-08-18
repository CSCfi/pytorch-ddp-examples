# Based on multiprocessing example from
# https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html

from datetime import datetime
import argparse
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from accelerate import Accelerator

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
    # Download dataset before setting up distributed processing since
    # torchvision dataset download logic is not multi-process safe ...
    if int(os.environ['RANK']) == 0:
        print('Downloading dataset (if needed) to', datadir)
        _ = MNIST(root=datadir, train=True, transform=transforms.ToTensor(),
                  download=True)

    accelerator = Accelerator()

    # some things are done only on global rank zero
    rank_zero = accelerator.is_main_process

    model = ConvNet()
    batch_size = 100

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    train_dataset = MNIST(root=datadir, train=True,
                          transform=transforms.ToTensor(), download=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=True)

    model, optimizer, train_loader = accelerator.prepare(model, optimizer,
                                                         train_loader)

    start = datetime.now()
    for epoch in range(args.epochs):
        tot_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)

            accelerator.backward(loss)
            optimizer.step()

            tot_loss += loss.item()

        if rank_zero:
            print('Epoch [{}/{}], average loss: {:.4f}'.format(
                epoch + 1,
                args.epochs,
                tot_loss / (i+1)))
    if rank_zero:
        print("Training completed in: " + str(datetime.now() - start))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=2, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loader workers')
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
