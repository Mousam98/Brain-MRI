# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 22:44:03 2021

@author: Mousam
"""


import numpy as np
import cv2  as cv

def findLargestComponent(img):
    _, labels, stats, _ = cv.connectedComponentsWithStats(img, 5)
    largest_labels = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
    longest_components = np.zeros(img.shape[:2], dtype = "uint8")
    longest_components[labels == largest_labels] = 255
    return longest_components