import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def img2df(img):
    h, w, _ = img.shape
    M = np.zeros((h * w, 3))
    for i in range(h):
        for j in range(w):
            M[i * w + j, :] = img[i, j, :]
    return pd.DataFrame(M, columns=["R", "G", "B"])

def clust_km(img, n_clusters=4):
    df = img2df(img)
    km = KMeans(n_clusters=n_clusters, random_state=44)
    km.fit(df)
    return km

def segmentation(img, km):
    img2 = np.copy(img)
    h, w, _ = img.shape
    for i in range(h):
        for j in range(w):
            idx = i * w + j
            label = km.labels_[idx]
            img2[i, j] = km.cluster_centers_[label]
    return img2
