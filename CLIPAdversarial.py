# %% md

# Applying CLIP model

"""
Code for generating adversarial examples for CLIP
Edit the parameters of the call of the function adversarial_sample
to change how the images are generated
"""

# magic for notebooks
'%load_ext autoreload'
'%autoreload 2'
'%matplotlib inline'

# %%

import os
import matplotlib.pyplot as plt
import clip
import torch
from torchvision.datasets import CIFAR100
import numpy as np
import PIL
from PIL import Image
from torchvision import transforms
from scipy.misc import imsave

# Had OpenMPI issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# %% md

## First test with constellation image

# %%

# Load constellations
from dataUtils import *

pictures = loadConstellations(pictureTypes=["original", "final_easy", "outline"])

# %%

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"  # force cpu
# Different architectures, choose one
model, preprocess = clip.load('ViT-B/32', device)  # Faster
# model, preprocess = clip.load('RN50x4', device) #Better but takes more time


# %%

# Download the dataset (not necessary, as only interested in classes)
# cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
# Coco = CocoDetection(root=os.path.expanduser("~/.cache"), download=True, train=False)

# %%

# Available classifications from different datasets
# combinedClasses.txt - CoCo + CIFAR100 (around 160 classes, removed duplicates)
# combinedClassesAdditional.txt -CoCo + CIFAR100 + classes about pictures which constellations are in set but
# not in either CoCo or CIFAR100 classes (around 180 classes, removed duplicates)
# cocoClasses.txt - CoCo (80 classes)
# cifarClasses.txt - Cifar100 (100 classes)
# imagenet_classes.txt - Imagenet classes (around 1000 classes)

chosenDataset = "combinedClassesAdditional.txt"
# Read the categories
with open(os.path.join("classes", chosenDataset), "r") as f:
    categories = [s.strip() for s in f.readlines()]

# %%

# Prepare the inputs
# image, class_id = cifar100[3637]
n = len(pictures.keys())


def adversarial_sample(model, text_inputs, labels, n_steps, step_size, start_im, start_im_strength=1.0, n=1,
                       start_im_flashback=0.0,
                       outfile=None, initial_noise=0.0, step_noise=0.0):
    label = labels[0]

    t_label = torch.from_numpy(label).to(device)
    im = start_im_strength * start_im + torch.rand(n, 3, 224, 224).to(device) * initial_noise
    im_noise = torch.randn_like(im).to(device)

    # Then refine the images
    for i in range(n_steps):
        im_noise.normal_()
        im = im + step_noise * im_noise
        im = im + start_im_flashback * start_im
        im.requires_grad_(requires_grad=True)
        print('\rStep %d/%d' % (i + 1, n_steps), end='')

        # Calculate features
        image_features = model.encode_image(im)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)

        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features_norm @ text_features.T).softmax(dim=-1)

        loss = ((similarity - t_label) ** 2).sum() / n
        im_grad = torch.autograd.grad([loss], [im])[0]

        im = im - step_size * im_grad
        im = im.detach()

        im = torch.clamp(im, -2, 2)

    with torch.no_grad():
        image_features = model.encode_image(im)
        text_features = model.encode_text(text_inputs)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    losses = ((similarity - t_label) ** 2).sum(dim=-1).detach().cpu().numpy()  # (1, 2, 3)
    best = np.argmin(losses, axis=0)

    if outfile is not None:
        output = im.detach().cpu().numpy()
        output = output.transpose((0, 2, 3, 1))
        output = output.reshape((-1, n, 224, 224, 3)).transpose((0, 2, 1, 3, 4)).reshape((-1, 224 * n, 3))
        imsave(outfile, output)
    return im[best].detach().cpu().numpy().transpose((1, 2, 0))

wanted_class = 95
print('Trying to fool for', categories[wanted_class])

#n = min(n, 6)
for i in range(n):
    objectId = list(pictures.keys())[i]
    constellationImage = pictures[objectId]["final_easy"][-1]
    outlineImage = pictures[objectId]["outline"][0]
    originalImage = pictures[objectId]["original"][0]
    print(objectId)
    # print(labels[i])
    # print(labels[i] in categories)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 8))
    axes[0].imshow(constellationImage)
    axes[1].imshow(outlineImage)
    axes[2].imshow(originalImage)
    plt.show()

    # for image_type in ["final_easy", "outline", "original"]:
    for image_type in ["final_easy"]:
        print(image_type)

        #    for image_type in ["outline", "final_easy"]:

        image = PIL.Image.fromarray(pictures[objectId][image_type][0])
        image_input = preprocess(image).unsqueeze(0).to(device)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in categories]).to(device)

        wanted = np.zeros((len(categories)))
        wanted[wanted_class] = 1

        adv_image_input = adversarial_sample(model, text_inputs, [wanted], 30, 400.0, start_im=image_input,
                                             start_im_flashback=0.0000, initial_noise=0.0, n=1,
                                             step_noise=0.001, start_im_strength=1.0,
                                             outfile=os.path.join('simulations', 'adversarial',
                                                                  'adversarial_%d.png' % i))

        image = PIL.Image.fromarray(adv_image_input.astype(np.uint8))
        image_input = preprocess(image).unsqueeze(0).to(device)
        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(5)

        # Print the result
        print("\nTop predictions for", image_type, ":\n")
        for value, index in zip(values, indices):
            print(f"{categories[index]:>16s}: {100 * value.item():.2f}%")

# %%
