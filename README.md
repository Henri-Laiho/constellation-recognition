# constellation-recognition

## Setup

1. Clone this repository
2. Merge the contents of this 7z archive into the cloned repository:
   https://drive.google.com/file/d/1d05SC3vCEqLf5mlnlw1_BmHRHdbdz0l4/view?usp=sharing
3. Install CLIP as instructed here https://github.com/openai/CLIP
4. Install PyTorch torch~=1.7.1 torchvision~=0.8.2 with the instruction at https://pytorch.org/
5. Install requirements from requirements.txt

## Structure

### Notebooks/Python files
* CLIP.ipynb - notebook for testing CLIP model
* dataUtils.py - functions for loading constellation images
* starUtils.py - functions for finding centres of stars from picture and drawing lines between them
* simulateSearchingKnownObjectFromStars.ipynb - notebook to simulate searching for a constellation in a way that the person knows what it is looking for. E.g. I'm looking for sunglasses from given set of stars

### Folders

#### classes
Folder containing text files containing list of classes to be used for Zero-shot learning gathered from CoCo and CIFAR100 datasets
* combinedClasses.txt - CoCo + CIFAR100 (around 160 classes, removed duplicates)
* combinedClassesAdditional.txt -CoCo + CIFAR100 + classes about pictures which constellations are in the given dataset but not in either CoCo or CIFAR100 classes (around 180 classes, removed duplicates)
* cocoClasses.txt - classes from CoCo (80 classes)
* cifarClasses.txt - classes from Cifar100 (100 classes)
* imagenet_classes.txt - Imagenet classes (around 1000 classes)

#### constellationImages
Folder containing two sets of constellation objects:
* limitedSet - around 50 pictures
* trainingSet - around 10 pictures

Each constellation object contains following images in separate subfolders:
* dotted
* final_easy
* final_hard
* original
* outline

#### simulations
Folder containing the results of conducted simulations for the construction of constellation from stars. Currently only has subfolder for simulateSearchingKnownObjectFromStars.
