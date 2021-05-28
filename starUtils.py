#Installation: pip install opencv-python
import cv2

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import copy

"""
Function for finding the coordinates of the stars in the constellation image

##Modified from source:
#https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/


constellationImg - image in numpy array (pixelsX, pixelsY, 3)
showPicture - Boolean for debugging, shows the constellation picture with centres

Returns:
    Coordinates of the centres of the stars in list 2D list

"""
def findStarCoordinates(constellationImg, showPicture=False):
    img = copy.deepcopy(constellationImg)
    starCoordinates = []
    # reading the image in grayscale mode
    # Convert image to grayscale
    grays = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    grays = grays.astype(np.uint8)
    # Set treshold to convert grayscale image to binary image
    th, threshed = cv2.threshold(grays, 60, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Find contours of stars
    contours = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    # For each contour find its center
    for contour in contours:
        moments = cv2.moments(contour)
        # Calculate the coordinates of contour
        if moments["m00"] != 0:
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
        else:
            # Gave division by zero error, workaround to it
            cX, cY = 0, 0

        starCoordinates += [(cX, cY)]

        # For testing, add centers to picture
        if (showPicture):
            cv2.circle(img, (cX, cY), 1, (255, 0, 0), -1)

    # Show the picture, when testing
    if (showPicture):
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return starCoordinates

"""
Function to draw white line between two points

Arguments:
    img - image as numpy array (pixelsX, pixelsY, 3)
    point1 - the starting point of line
    point2 - the ending point of line

Returns
    Image with line drawn between two points
"""
def drawLine(img, point1, point2, thickness=1):
    lineImage = copy.deepcopy(img)
    pts = np.array([point1, point2])
    pts = pts.reshape((-1, 1, 2))
    lineImage = cv2.polylines(lineImage, [pts], False, (255, 255, 255), thickness=thickness, lineType=cv2.LINE_AA)
    return lineImage