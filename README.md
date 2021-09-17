# PyTorch DDP examples for CSC Supercomputers

[PyTorch Distributed][1], and in particular [Distributed Data-Parallel
(DDP)][2], offers a nice way of running multi-GPU and multi-node PyTorch jobs.
Unfortunately, in this instance, the official PyTorch documentation and usage
examples are sadly out-of-date with often conflicting and confusing advice
given.

To make usage of DDP on CSC's Supercomputers easier, we have created a set of examples
on how to run simple DDP jobs on the cluster. All example train a simple CNN on
MNIST.

## Multi-GPU, single-node

The simplest case is using all four GPUs on a single node on Puhti.

```bash
sbatch run-gpu4-dist.sh mnist_ddp.py --epochs=100
```

## Multi-GPU, multi-node

Example using two nodes, four GPUs on each giving a total of 8 GPUs (again, on Puhti):

```bash
sbatch run-gpu8-dist.sh mnist_ddp.py --epochs=100
```

## PyTorch Lightning examples

Multi-GPU and multi-node jobs are even easier with [PyTorch Lightning][3].

Four GPUs on single node on Puhti:

```bash
sbatch run-gpu4.sh mnist_lightning_ddp.py --gpus=4 --epochs=100
```

**NOTE:** Lightning seems to work a bit slower on Puhti than pure PyTorch at the
moment. We are still investigating this.


Two nodes, 8 GPUs in total on Puhti:

```bash
sbatch run-gpu8-dist.sh mnist_lightning_ddp.py --gpus=4 --nodes=2 --epochs=100
```

**NOTE:** Multi-node Lightning still seems a bit unstable on Puhti. We are
investigating.

## DeepSpeed examples

[DeepSpeed][4] was installed on Puhti like this:

```bash
module purge
module load pytorch/1.9
TORCH_CUDA_ARCH_LIST="7.0" DS_BUILD_OPS=1 DS_BUILD_AIO=0 DS_BUILD_TRANSFORMER_INFERENCE=0 \
  pip install -vv --user deepspeed
```

Note that we are pre-compiling all of the DeepSpeed C++/CUDA ops, except a few
that had problems. We'll have to get back to those later if they are needed.

Single-node with four GPUs (Puhti):

```bash
sbatch run-gpu4-deepspeed.sh mnist_ddp_deepspeed.py --epochs=100 --deepspeed --deepspeed_config ds_config.json
```

Two nodes, 8 GPUs in total (Puhti):

```bash
sbatch run-gpu8.sh mnist_ddp_deepspeed.py --epochs=100 --deepspeed --deepspeed_config ds_config.json
```


[1]: https://pytorch.org/tutorials/beginner/dist_overview.html
[2]: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
[3]: https://www.pytorchlightning.ai/
[4]: https://www.deepspeed.ai/
