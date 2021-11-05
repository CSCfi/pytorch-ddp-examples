# Based on multiprocessing example from
# https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html

from datetime import datetime
import argparse
import os
import torch
import torch.distributed as dist
import torch.nn as nn

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import deepspeed
from datetime import timedelta

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models


def train(args):
    num_epochs = args.epochs
    local_rank = args.local_rank
    if local_rank == -1:
        local_rank = int(os.environ.get('PMIX_RANK', -1))

    deepspeed.init_distributed(timeout=timedelta(minutes=5))

    torch.manual_seed(0)

    # Set up standard model.
    if local_rank == 0:
        print('Using {} model'.format(args.model))
    model = getattr(models, args.model)()
    model = model.cuda()
    
    criterion = nn.CrossEntropyLoss().cuda()

    traindir = os.path.join(args.datadir, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

    model_engine, optimizer, train_loader, __ = deepspeed.initialize(
        args=args, model=model, model_parameters=model.parameters(),
        training_data=train_dataset)

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            images = data[0].to(model_engine.local_rank)
            labels = data[1].to(model_engine.local_rank)

            outputs = model_engine(images)
            loss = criterion(outputs, labels)

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
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--model', type=str, default='resnet50',
                        help='model to benchmark')
    parser.add_argument('--datadir', type=str,
                        help='Data directory')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')

    parser = deepspeed.add_config_arguments(parser)
    
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
