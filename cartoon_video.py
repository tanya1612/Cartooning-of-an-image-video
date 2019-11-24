import cv2
import matplotlib.pyplot as plt
import numpy
from sklearn.cluster import KMeans

cap = cv2.VideoCapture(0)

while True:
    ret, img_rgb = cap.read()
    cv2.imshow('frame', img_rgb)

    colors = 32
    img_rgbs = img_rgb.reshape((-1, 3))
    kmeans = KMeans(n_clusters = colors, random_state = 67).fit(img_rgbs)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    less_colors = centers[labels].reshape(img_rgb.shape).astype('uint8')

    for _ in range(7):
      filtered = cv2.bilateralFilter(less_colors, 9, 9, 7)

    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    edge = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7,2 )

    edges = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(filtered, edges)

    cv2.imshow('cartoon', cartoon)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.waitKey(0)
cv2.destroyAllWindows
