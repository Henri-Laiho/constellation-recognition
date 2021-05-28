from torchvision.datasets import CIFAR100
import os
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=False, train=False)
for value in cifar100.classes:
    print(value)