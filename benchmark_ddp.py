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

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models


def train(args):
    local_rank = int(os.environ['LOCAL_RANK'])
    if local_rank == 0:
        print('Using PyTorch version:', torch.__version__)
        print(torch.__config__.show())

    dist.init_process_group(backend='nccl')

    torch.manual_seed(0)
    torch.cuda.set_device(local_rank)

    # Set up standard model.
    if local_rank == 0:
        print('Using {} model'.format(args.model))
    model = getattr(models, args.model)()
    model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    model = DistributedDataParallel(model, device_ids=[local_rank])

    traindir = os.path.join(args.datadir, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batchsize,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=True, sampler=train_sampler)

    start = datetime.now()
    total_step = args.steps if args.steps is not None else len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0 and local_rank == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1,
                    args.epochs,
                    i + 1,
                    total_step,
                    loss.item()))
            if args.steps is not None and i >= args.steps:
                break
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
    parser.add_argument('-b', '--batchsize', type=int, default=32,
                        help='Batch size')
    parser.add_argument('-j', '--workers', type=int, default=10,
                        help='Number of data loader workers')
    parser.add_argument('--steps', type=int, required=False,
                        help='Maxium number of training steps')
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
