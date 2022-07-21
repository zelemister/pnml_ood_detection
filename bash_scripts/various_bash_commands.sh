#!/bin/bash

cd ../src || exit
#python main_execute_method.py method=baseline model=vicreg trainset=cifar10 batch_size=4096
python vicreg_evaluate.py --data-dir ../data --pretrained ../models/resnet50.pth --exp-dir ../outputs/VICReg_Training --lr-head 0.02 --dataset svhn