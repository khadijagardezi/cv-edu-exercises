import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load Image
image = cv2.imread('img0.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load Pre-trained Feature Extractor (ResNet50)
model = models.resnet50(pretrained=True)
model.eval()

# Define transformation for the input image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Transform the image
input_tensor = transform(image).unsqueeze(0)

# Extract Features
with torch.no_grad():
    features = model(input_tensor)

# KMeans Clustering for Image Segmentation
pixels = image.reshape(-1, 3)
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(pixels)
segmented_image = kmeans.labels_.reshape(image.shape[0], image.shape[1])

# Display Segmented Image
plt.imshow(segmented_image)
plt.axis('off')
plt.show()
