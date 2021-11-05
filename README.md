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
TORCH_CUDA_ARCH_LIST="7.0;8.0" DS_BUILD_OPS=1 DS_BUILD_AIO=0 DS_BUILD_TRANSFORMER_INFERENCE=0 \
  pip install -vv --user deepspeed
```

Note that we are pre-compiling all of the DeepSpeed C++/CUDA ops, except a few
that had problems. We'll have to get back to those later if they are needed.

Single-node with four GPUs (Puhti):

```bash
sbatch run-gpu4-deepspeed.sh mnist_ddp_deepspeed.py --epochs=100 --deepspeed --deepspeed_config ds_config.json
```

Here we are using Slurm to launch a single process which uses DeepSpeed's
launcher to launch four processes (one for each GPU).

Two nodes, 8 GPUs in total (Puhti):

```bash
sbatch run-gpu8.sh mnist_ddp_deepspeed.py --epochs=100 --deepspeed --deepspeed_config ds_config.json
```

Note that we are using Slurm's `srun` to launch four processess on each node
(one per GPU), and instead of DeepSpeed's launcher we are relying on MPI to
provide it the information it needs to communicate between all the processes.
Also remember to install `mpi4py` package, for example with `pip install --user mpi4py`.

Finally, we hade to make a small change to the DeepSpeed source code as it would
by default use the wrong IP address (and thus the wrong interface) to connect to
the master node (causing the initialization phase to get stuck and time out
eventually):

```bash
$ diff ~/.local/lib/python3.8/site-packages/deepspeed/utils/distributed.py{.old,}
66c66
<         hostname_cmd = ["hostname -I"]
---
>         hostname_cmd = ["hostname -s"]
```

In the original code `hostname -I` returned a list of IPs, one for each network
interface. DeepSpeed would simply pick the first one, which unfortunately was
for the Ethernet interface, which seems to be blocked. By using `hostname -s` we
get the short hostname, and using that it seems to connect over InfiniBand
instead of Ethernet, which works.

## Benchmark codes

Finally, we have some "benchmarking" scripts which run larger training jobs.
Currently, these are for Mahti only.

ResNet training with ImageNet data, PyTorch DDP with 1, 4 or 8 GPUs:

```
run-gpu1-benchmark-ddp.sh
run-gpu4-benchmark-ddp.sh
run-gpu8-benchmark-ddp.sh
```

ResNet training with ImageNet data, PyTorch DeepSpeed with 4 or 8 GPUs:

```
run-gpu4-benchmark-deepspeed.sh
run-gpu8-benchmark-deepspeed.sh
```

HuggingFace Transformers fine-tuning GPT-2:

```
run-gpu1-benchmark-transformers.sh
run-gpu4-benchmark-transformers.sh
run-gpu8-benchmark-transformers.sh
```

Note that the Transformers benchmark requires checking out the right version of
the official repository, for example:

```bash
git clone -b v4.11.3 https://github.com/huggingface/transformers/
```

[1]: https://pytorch.org/tutorials/beginner/dist_overview.html
[2]: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
[3]: https://www.pytorchlightning.ai/
[4]: https://www.deepspeed.ai/
