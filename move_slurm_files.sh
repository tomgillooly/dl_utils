#!/usr/bin/env bash

for file in $(ls $(pwd)/checkpoints/*/opts*)
do
  dir=$(dirname $file)
  slurm_id=$(awk -F ":" '/slurm_job_id/{print $2}' $file)
  test $slurm_id == 'None' && continue
  scontrol -dd show job $slurm_id 1>/dev/null 2>/dev/null || (test -e slurm-"$slurm_id".out && mv slurm-"$slurm_id".out $dir)
  scontrol -dd show job $slurm_id 1>/dev/null 2>/dev/null && test -e slurm-"$slurm_id".out &&  cp slurm-"$slurm_id".out $dir
done