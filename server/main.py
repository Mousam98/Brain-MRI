import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mat
import cv2 as cv
from PIL import Image
from clusters import kmeans, fcm
from OpenCVStatistics import findLargestComponent

image = cv.imread("F:/MyML/UCI_Datasets/Brain_Tumor_Detection/Sample_Test/image5.jpg")

def rescale(img, scale):
    (height, width) = img.shape[:2]
    if abs(height-width) >= 150: 
        height, width = int(height*scale), int(width*scale)
    dimensions = (width, height)
    return cv.resize(img, dimensions, interpolation = cv.INTER_AREA)


'''rescaling image'''
image = rescale(image, 0.50)

grayscaled_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
plt.imshow(grayscaled_image)
plt.show()

'''filtering image'''
filtered_image = cv.bilateralFilter(grayscaled_image, 5, 15, 20)

'''thresholding image'''
thres, threshold = cv.threshold(filtered_image, 10, 255, cv.THRESH_BINARY)
#cv.imshow("thresolded Image", threshold)
plt.imshow(threshold)
plt.show()

'''extracting brain portion'''
extracted_image = findLargestComponent(threshold)
#cv.imshow("Longest Component", extracted_image)
plt.imshow(extracted_image)
plt.show()

'''applying mathematical morphology'''
kernel = np.ones((28,28), dtype = "uint8")
morpho_image = cv.morphologyEx(extracted_image, cv.MORPH_CLOSE, kernel)
'''print(np.count_nonzero(morpho_image == 1))'''

masked_image = cv.bitwise_and(filtered_image, filtered_image, mask = morpho_image)
kmeans_input = masked_image.reshape(masked_image.shape[0]*masked_image.shape[1], 1)
kmeans_input = np.float32(kmeans_input)

'''applying kmeans'''
n_clusters = 5
kmeans_image = kmeans(kmeans_input, n_clusters)
kmeans_image = kmeans_image.reshape(masked_image.shape)

'''applying fcm'''
fcm_image = fcm(kmeans_image, n_clusters)

plt.imshow(kmeans_image)
plt.show()
plt.imshow(fcm_image)
plt.show()

mat.image.imsave("F:/MyML/UCI_Datasets/Brain_Tumor_Detection/Test Outputs/image5.jpg", kmeans_image)
