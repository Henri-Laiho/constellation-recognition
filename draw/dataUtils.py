#### Functions to load data

from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from collections import defaultdict

"""
Function for loading the constellation images (dots on black background) with original pictures.
Each constellation has multiple pictures with or without added noise.

Arguments:
 selectedDataset - can be "limitedSet" - handpicked ~40 constellation
                          "fullSet" - all the constellations
 pictureTypes - list defining what sort of constellations are loaded
                "dotted" - only constellation without noise
                "final_easy" - constellation with little noise
                "final_hard" - constellations with more noise
                "outline" - dots with added lines between them

 pictureSize - Size to resize images to after they are read from disk. Defaults to (256, 256).

Returns:
  4 numpy arrays
  constellationPictures - the pictures of constellations
  constellationLabels - object IDs of constellations
  originalPictures - the original pictures from which the constellations are created
  originalLabels - the object IDs of original pictures

"""


def loadThingsOutlines(dataset, pictureTypes, pictureSize, colorMode):
    pictures = defaultdict(lambda: defaultdict(list))
    objects = os.listdir(os.path.join(".", "constellationImages", dataset))
    i = 0
    for object in objects:
        pictureNames = os.listdir(os.path.join(".", "constellationImages", dataset, object))
        for pictureName in pictureNames:
            i += 1
            if i % 1000 == 0:
                print("Loading images", i)
            img = image.load_img(os.path.join(".", "constellationImages", dataset, object, pictureName), target_size=pictureSize, color_mode=colorMode)
            # Convert to np array and add to list
            pictures[object][pictureTypes[0]].append(np.array(img))
        pictures[object][pictureTypes[0]] = np.array(pictures[object][pictureTypes[0]])
    return pictures

def loadConstellations(selectedDataset="limitedSet", pictureTypes=["original", "final_easy"], pictureSize=(256, 256), colorMode="rgb"):
    if "things" in selectedDataset.lower():
        return loadThingsOutlines(selectedDataset, pictureTypes, pictureSize, colorMode)

    pictures = defaultdict(lambda: defaultdict(list))

    constellationObjectIds = os.listdir(os.path.join(".", "constellationImages", selectedDataset))
    # For each constellation load all the picture from given types of picture
    # Store also the object ID for each loaded picture
    # Also load the original picture and store its object ID for later comparison
    for constellationObjectId in constellationObjectIds:
        for pictureType in pictureTypes:
            constellationPictureNames = os.listdir(os.path.join(".", "constellationImages", selectedDataset,
                                                                constellationObjectId, pictureType))
            for constellationPictureName in constellationPictureNames:
                # Load image
                
                img = image.load_img(os.path.join(".", "constellationImages",
                                                  selectedDataset, constellationObjectId,
                                                  pictureType, constellationPictureName),
                                     target_size=pictureSize, color_mode=colorMode)
                # Convert to np array and add to list
                pictures[constellationObjectId][pictureType].append(np.array(img))
            pictures[constellationObjectId][pictureType] = np.array(pictures[constellationObjectId][pictureType])

    return pictures


"""
Function to find the original picture of chosen constellation by object ID

Arguments:
objectID - constellations ID
originalPictures - list containing the original pictures
originalLabels - list containing the IDs of original pictures, in same order as originalPictures

"""
def getOriginalForConstellation(objectID, pictures):
    return pictures[objectID]["original"]