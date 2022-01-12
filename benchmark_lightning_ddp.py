from datetime import datetime
from torch.utils.data import DataLoader
from torchvision import models
import argparse
import os
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.plugins import DDPPlugin


class TorchvisionModel(pl.LightningModule):
    def __init__(self, model_name):
        super().__init__()
        self.model = getattr(models, model_name)()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), 1e-4)
        return optimizer


def train(args):
    print('Using PyTorch version:', torch.__version__)
    print(torch.__config__.show())

    model = TorchvisionModel(args.model)

    traindir = os.path.join(args.datadir, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batchsize,
                              num_workers=args.workers, pin_memory=True)

    trainer = pl.Trainer(gpus=args.gpus,
                         num_nodes=args.nodes,
                         max_epochs=args.epochs,
                         accelerator='gpu',
                         strategy=DDPPlugin(find_unused_parameters=False))
#                         strategy=DDPStrategy(find_unused_parameters=False))
#                         strategy='ddp')

    t0 = datetime.now()
    trainer.fit(model, train_loader)
    dt = datetime.now() - t0
    print('Training took {}'.format(dt))

    trainer.save_checkpoint("benchmark_lightning_model.ckpt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default=1, type=int, metavar='N',
                        help='number of GPUs per node')
    parser.add_argument('--nodes', default=1, type=int, metavar='N',
                        help='number of nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='maximum number of epochs to run')

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
