
from torchvision.datasets import CIFAR100
import os
import json

combinedClasses = []
### Get CIFAR classes
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=False, train=False)
combinedClasses+=cifar100.classes
### Get CoCo classes

json_file = './CoCoAnnotations/instances_val2017.json'
cocoClasses = []
with open(json_file,'r') as COCO:
    js = json.loads(COCO.read())
    categories = js["categories"]
    for category in categories:
        ### Augment two list of classes by leaving out the ones that already exist in other Dataset
        if category["name"] not in combinedClasses:
            combinedClasses.append(category["name"])
###Write file
file = open("combinedClasses.txt", "w+")
for value in combinedClasses:
    file.write(value+"\n")
file.close()



