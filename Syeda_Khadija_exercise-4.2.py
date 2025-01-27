import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

image = cv2.imread('dog.jpeg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_rgb = cv2.resize(image_rgb, (400, 300))

pixels = image_rgb.reshape(-1, 3)
pixels = StandardScaler().fit_transform(pixels)  

# KMeans Clustering
kmeans = MiniBatchKMeans(n_clusters=2, random_state=0, batch_size=1000)
kmeans_labels = kmeans.fit_predict(pixels)
kmeans_result = kmeans_labels.reshape(image_rgb.shape[0], image_rgb.shape[1])

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.3, min_samples=10)
dbscan_labels = dbscan.fit_predict(pixels)

dbscan_result = dbscan_labels.reshape(image_rgb.shape[0], image_rgb.shape[1])
dbscan_result = np.where(dbscan_result == -1, 0, dbscan_result)

#  Gaussian Mixture Model
gmm = GaussianMixture(n_components=2, random_state=0)
gmm_labels = gmm.fit_predict(pixels)
gmm_result = gmm_labels.reshape(image_rgb.shape[0], image_rgb.shape[1])

fig, ax = plt.subplots(1, 4, figsize=(20, 10))
ax[0].imshow(image_rgb)
ax[0].set_title('Original Image')
ax[1].imshow(kmeans_result)
ax[1].set_title('KMeans Clustering')
ax[2].imshow(dbscan_result)
ax[2].set_title('DBSCAN Clustering')
ax[3].imshow(gmm_result)
ax[3].set_title('GMM Clustering')

for a in ax:
    a.axis('off')

plt.show()
