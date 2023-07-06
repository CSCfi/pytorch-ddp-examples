# Based on multiprocessing example from
# https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html

from datetime import datetime
import argparse
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import mlflow

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


def train(num_epochs):
    dist.init_process_group(backend='nccl')

    torch.manual_seed(0)
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    verbose = dist.get_rank() == 0  # print only on global_rank==0
    if verbose:
        mlflow.set_tracking_uri("/scratch/project_2001659/mvsjober/mlruns")
        #mlflow.set_tracking_uri("sqlite:////scratch/project_2001659/mvsjober/mlruns.db")
        #mlflow.set_tracking_uri('https://mats-mlflow2.rahtiapp.fi/')

        mlflow.start_run(run_name=os.getenv("SLURM_JOB_ID"))

    model = ConvNet().cuda()
    batch_size = 100

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    model = DistributedDataParallel(model, device_ids=[local_rank])

    train_dataset = MNIST(root='./data', train=True,
                          transform=transforms.ToTensor(), download=True)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True,
                              sampler=train_sampler)

    start = datetime.now()
    for epoch in range(num_epochs):
        tot_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tot_loss += loss.item()

        if verbose:
            tot_loss = tot_loss / (i+1)
            mlflow.log_metric("loss", tot_loss)
            
            print('Epoch [{}/{}], average loss: {:.4f}'.format(
                epoch + 1,
                num_epochs,
                tot_loss))
    if verbose:
        print("Training completed in: " + str(datetime.now() - start))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()

    train(args.epochs)


if __name__ == '__main__':
    main()
