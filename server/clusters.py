# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 22:39:39 2021

@author: Mousam
"""

import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
from skfuzzy.cluster import cmeans

def kmeans(img, n_clusters):
    model = KMeans(n_clusters = n_clusters, 
                   init = "k-means++", n_init= 10, max_iter = 10, 
                   random_state = 0, 
                   n_jobs = -1, algorithm = "full")
  
    model.fit(img)
    centers = model.cluster_centers_
    labels = model.labels_
    kmeans_output = centers[labels]
    return kmeans_output
    


def fcm(img, n_clusters):
    M = 2
    epsilon = 0.01
    k = np.ones((5,5), dtype = "uint8")
    img = cv.blur(img, (15, 15))
    flat = img.reshape((1, -1))
    c, u = cmeans(flat, n_clusters, M, epsilon, 50)[0:2]
    tumor_index = np.argmax(c, axis=0)
    defuz = np.argmax(u, axis=0)
    mask = np.full(defuz.shape[0], 0, dtype=np.uint16)
    mask[defuz == tumor_index] = 1
    mask = mask.reshape(img.shape)
    mask = cv.erode(mask, k, iterations=1)
    mask = cv.dilate(mask, k, iterations=1)
    return mask
