#Este código fue hecho en coolaboracion con José Angel Ramírez Ramírez
import cv2
import numpy as np
import os

def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        cnt = contours[0]
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            compactness = 0
        else:
            compactness = 4 * np.pi * area / (perimeter ** 2)
        if area == 0:
            circularity = 0
        else:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
        _, _, w, h = cv2.boundingRect(cnt)
        if w == 0 or h == 0:
            excentricity = 0
        else:
            excentricity = np.sqrt(1 - ((h/w) ** 2))
        # Agregar histograma de color como característica
        color_hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        color_hist = cv2.normalize(color_hist, color_hist).flatten()
        return [area, perimeter, compactness, circularity, excentricity] + list(color_hist)
    else:
        return [0, 0, 0, 0, 0] + [0] * 512


# Leer imágenes y extraer características
features = []
labels = []
for folder in ['1', '2', '3', '4', '5']:
    for filename in os.listdir(os.path.join("/home/fred/Vision_computadora/", folder)):
        if not filename.endswith('.ppm'):
            continue
        img = cv2.imread(os.path.join("/home/fred/Vision_computadora/", folder, filename))
        feature_vector = extract_features(img)
        if not np.isnan(np.sum(feature_vector)):
            features.append(feature_vector)
            labels.append(folder)

# Entrenar modelo de clasificación
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy: {acc:.2f}')
