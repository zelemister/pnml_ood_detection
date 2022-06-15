import torch
dir = "../outputs/VICReg_Training/checkpoint.pth"
m = torch.load(dir)["model"]
torch.save(m, "../models/vicreg_cifar10.pth")