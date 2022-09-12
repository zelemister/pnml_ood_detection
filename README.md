# Run to code

Install requirements
```bash
# Create env
conda create -n pnml_ood python=3.8.0 --yes
conda activate pnml_ood

# Install pip for fallback
conda install --yes pip

# Pytorch with GPU
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch --yes

# All other: Install with conda. If package installation fails, install with pip.
while read requirement; do conda install --yes $requirement || pip install $requirement; done < requirements.txt 
```

Download data and models
```bash
# Download OOD data
cd bash_scripts
chmod 777 ./download_data.sh
./download_data.sh

# Download pretrained models
chmod 777 ./download_models.sh
./download_models.sh
chmod 777 ./download_vicreg.sh
./download_vicreg.sh


```
## Pretrain VICReg
```bash

#run those commands to pretrain the vicreg on the datasets cifar10, cifar100
python vicreg_evaluate.py --data-dir ../data --pretrained ../models/resnet50.pth --exp-dir ../outputs/VICReg_Training --weights finetune --train-perc 10 --epochs 20 --lr-backbone 0.03 --lr-classifier 0.08 --weight-decay 0 --dataset cifar10
python vicreg_evaluate.py --data-dir ../data --pretrained ../models/resnet50.pth --exp-dir ../outputs/VICReg_Training --weights finetune --train-perc 10 --epochs 20 --lr-backbone 0.03 --lr-classifier 0.08 --weight-decay 0 --dataset cifar100
```


## Execute methods
Execute a single method. Our pNML method runs on-top of the executed method
```bash
cd src
python main_execute_method.py model= trainset=cifar100 method=baseline batch_size=512
python main_execute_method.py model= trainset=cifar10 method=baseline batch_size=512


```


Create paper's tables
```bash
cd src
python main_create_tables.py
```
