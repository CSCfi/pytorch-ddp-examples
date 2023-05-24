# PyTorch multi-GPU and multi-node examples for CSC's supercomputers

[PyTorch distributed][pytorch_dist] and in particular
`DistributedDataParallel` (DDP), offers a nice way of running
multi-GPU and multi-node PyTorch jobs. Unfortunately, the PyTorch
documentation has been a bit lacking in this area, and examples found
online can often be out-of-date.

To make usage of DDP on CSC's Supercomputers easier, we have created a
set of examples on how to run simple DDP jobs on the cluster. Included
are also examples with other frameworks, such as [PyTorch
Lightning][lightning] and [DeepSpeed][deepspeed].

All examples train a simple CNN on MNIST. Scripts have been provided
for the Puhti supercomputer, but can be used on other systems with
minor modifications.

For larger examples, see also our [Machine learning benchmarks
repository](https://github.com/mvsjober/ml-benchmarks).

Finally, you might also be interested in [CSC's machine learning
guide](https://docs.csc.fi/support/tutorials/ml-guide/) and in
particular the section on [Multi-GPU and multi-node
machinelearing](https://docs.csc.fi/support/tutorials/ml-multi/).


## Multi-GPU, single-node

The simplest case is using all four GPUs on a single node on Puhti.

```bash
sbatch run-ddp-gpu4.sh
```

## Multi-GPU, multi-node

Example using two nodes, four GPUs on each giving a total of 8 GPUs (again, on Puhti):

```bash
sbatch run-ddp-gpu8.sh
```


## PyTorch Lightning examples

Multi-GPU and multi-node jobs are even easier with [PyTorch
Lightning][lightning]. The [official PyTorch Lightning now has a
relatively good Slurm
documentation](https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html?highlight=slurm),
although it has to be modified a bit for Puhti.

Four GPUs on single node on Puhti:

```bash
sbatch run-lightning-gpu4.sh
```

Two nodes, 8 GPUs in total on Puhti:

```bash
sbatch run-lightning-gpu8.sh
```

## DeepSpeed examples

[DeepSpeed][deepspeed] should work on Puhti and Mahti with the
[PyTorch module](https://docs.csc.fi/apps/pytorch/) (from version 1.10
onwards).

Single-node with four GPUs (Puhti):

```bash
sbatch run-deepspeed-gpu4.sh
```

Here we are using Slurm to launch a single process which uses DeepSpeed's
launcher to launch four processes (one for each GPU).

Two nodes, 8 GPUs in total (Puhti):

```bash
sbatch run-deepspeed-gpu8.sh
```

Note that we are using Slurm's `srun` to launch four processess on each node
(one per GPU), and instead of DeepSpeed's launcher we are relying on MPI to
provide it the information it needs to communicate between all the processes.


[pytorch_dist]: https://pytorch.org/tutorials/beginner/dist_overview.html
[ddp]: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
[lightning]: https://www.pytorchlightning.ai/
[deepspeed]: https://www.deepspeed.ai/
