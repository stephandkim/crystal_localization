import numpy as np
# import cupy as np
import cv2

def cluster(img, K):

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    attempts = 10
    ret, label, center=cv2.kmeans(img.vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

    img.label = label
    img.center = np.uint8(center)
    res = img.center[label.flatten()]
    img.clustered = res.reshape((img.raw.shape))

def cluster_loss(img):

    sum_array = np.zeros(img.center.shape[0])
    for idx, label in enumerate(img.label):
        centroid = img.center[label[0]]

        sum_array[label[0]] += np.linalg.norm(centroid - img.vectorized[idx])

    return sum(sum_array)


def cluster_analysis(img, k_range_list):
    loss_list = []
    for k in k_range_list:
        cluster(img, k)
        loss_list.append(cluster_loss(img))

    return loss_list