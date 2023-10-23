from datetime import datetime
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import multiprocessing as mp
import torch
import torch.nn as nn
import torchvision.transforms as transforms


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


def train(batch_size):
    num_epochs = 100

    torch.manual_seed(0)
    verbose = True

    model = ConvNet().cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    train_dataset = MNIST(root='./data', train=True,
                          transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

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
            print('Epoch [{}/{}], batch_size={} average loss: {:.4f}'.format(
                epoch + 1,
                num_epochs,
                batch_size,
                tot_loss / (i+1)))
    if verbose:
        print("Training completed in: " + str(datetime.now() - start))


if __name__ == '__main__':
    bs_list = [16, 32, 64, 128]
    num_processes = 4

    with mp.Pool(processes=num_processes) as pool:
        pool.map(train, bs_list)
