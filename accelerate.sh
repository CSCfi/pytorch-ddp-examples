#!/bin/bash

set -x
accelerate launch --machine_rank=$SLURM_NODEID $*
