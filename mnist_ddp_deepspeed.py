# Based on multiprocessing example from
# https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html

from datetime import datetime
import argparse
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
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
    #local_rank = int(os.environ['LOCAL_RANK'])
    local_rank = args.local_rank

    #dist.init_process_group(backend='nccl')
    print("Running init_distributed")
    deepspeed.init_distributed(timeout=timedelta(minutes=5))
    print("Done with init_distributed")

    torch.manual_seed(0)
    # torch.cuda.set_device(local_rank)
    model = ConvNet() #.cuda()
    batch_size = 100

    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    #model = DistributedDataParallel(model, device_ids=[local_rank])

    train_dataset = MNIST(root='./data', train=True,
                          transform=transforms.ToTensor(), download=True)
    # train_sampler = DistributedSampler(train_dataset)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
    #                           shuffle=False, num_workers=0, pin_memory=True,
    #                           sampler=train_sampler)

    print("Running initialize")
    model_engine, optimizer, train_loader, __ = deepspeed.initialize(
        args=args, model=model, model_parameters=model.parameters(),
        training_data=train_dataset)
    print("Done with initialize")
    
    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            images = data[0].to(model_engine.local_rank)
            labels = data[1].to(model_engine.local_rank)

            outputs = model_engine(images)
            loss = criterion(outputs, labels)

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            model_engine.backward(loss)
            model_engine.step()
            
            if (i + 1) % 100 == 0 and local_rank == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1,
                    num_epochs,
                    i + 1,
                    total_step,
                    loss.item()))
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
