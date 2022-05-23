import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

jpgs, npys = set(), set()
unconverted_images = []

for f in os.listdir('../images'):
    if f[-4:] == '.jpg':
        jpgs.add(f[:-4])
    else:
        npys.add(f[:-4])

for jpg in jpgs:
    if jpg in npys:
        continue
    else:
        unconverted_images.append(jpg)
        
def convert_and_save(img_name):
    img = cv2.imread('../images/' + img_name + '.jpg', cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    vectorized = np.float32(img.reshape((-1,3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    K = 2
    attempts = 10
    ret, labels, center = cv2.kmeans(
                                    vectorized,
                                    K, 
                                    None, 
                                    criteria, 
                                    attempts, 
                                    cv2.KMEANS_PP_CENTERS
                                    )

    for idx in range(len(labels)):
        if labels[idx] != 0:
            labels[idx] = 1

    labels = labels.reshape((img.shape[0], img.shape[1]))        
    np.save('../images/' + img_name, labels)
    
for img_name in unconverted_images:
    convert_and_save(img_name)
