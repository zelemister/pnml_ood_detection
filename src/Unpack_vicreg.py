import torch
dir = "../outputs/VICReg_Training/Cifar100/checkpoint.pth"
m = torch.load(dir)["model"]
torch.save(m, "../models/vicreg_cifar100.pth")