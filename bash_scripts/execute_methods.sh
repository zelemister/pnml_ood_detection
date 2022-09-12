#!/bin/bash

declare -a trainsets=("cifar10" "cifar100" "svhn")
declare -a methods="baseline"
declare -a models="vicreg"

cd ../src || exit

for trainset in ${trainsets[@]}; do
    echo $method $model $trainset

      batch_size=512
      python main_execute_method.py method=$method model=$model trainset=$trainset batch_size=$batch_size
done