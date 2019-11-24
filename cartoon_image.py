import cv2
import matplotlib.pyplot as plt
import numpy
from sklearn.cluster import KMeans

img = cv2.imread("/content/girl.jpg")
img_rgb = img

colors = 32
img_rgbs = img_rgb.reshape((-1, 3))
kmeans = KMeans(n_clusters = colors, random_state = 67).fit(img_rgbs)
labels = kmeans.labels_
centers = kmeans.cluster_centers_
less_colors = centers[labels].reshape(img_rgb.shape).astype('uint8')

for _ in range(7):
  filtered = cv2.bilateralFilter(less_colors, 9, 100, 100)

gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 3)
edge = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7,5 )

edges = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
cartoon = cv2.bitwise_and(filtered, edges)

cv2_imshow(img)
cv2_imshow(blur)
cv2_imshow(edges)
cv2_imshow(filtered)
cv2_imshow(cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows
