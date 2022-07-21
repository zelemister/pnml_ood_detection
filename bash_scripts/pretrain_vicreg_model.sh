#!/bin/bash

declare -a dataset = "cifar10"

cd ../src || exit
python vicreg_evaluate.py --data-dir ../data --pretrained ../models/resnet50.pth --exp-dir ../outputs/VICReg_Training --weights finetune --train-perc 10 --epochs 20 --lr-backbone 0.03 --lr-classifier 0.08 --weight-decay 0 --dataset dataset

#python vicreg_evaluate.py --data-dir ../data --pretrained ../models/resnet50.pth --exp-dir ../outputs/VICReg_Training/CIFAR10 --weights finetune --train-percent 10 --epochs 20 --lr-backbone 0.03 --lr-head 0.08 --weight-decay 0 --dataset cifar10