# PyTorch DDP examples for CSC Supercomputers

[PyTorch Distributed][1], and in particular [Distributed Data-Parallel
(DDP)][2], offers a nice way of running multi-GPU and multi-node PyTorch jobs.
Unfortunately, in this instance, the official PyTorch documentation and usage
examples are sadly out-of-date with often conflicting and confusing advice
given.

To make usage of DDP on CSC's Supercomputers, we have created a set of example
on how to run simple DDP jobs on the cluster. All example train a simple CNN on
MNIST.

## Multi-GPU, single-node

The simplest case is using all four GPUs on a single node.

```bash
sbatch run-gpu4-dist.sh mnist_ddp.py --epochs=100
```

## Multi-GPU, multi-node

Example using two nodes, four GPUs on each giving a total of 8 GPUs:

```bash
sbatch run-gpu8-dist.sh mnist_ddp.py --epochs=100
```
