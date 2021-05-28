# constellation-recognition

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
