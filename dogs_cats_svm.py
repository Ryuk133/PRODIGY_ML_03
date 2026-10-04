import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data_path = "train_transformed"

images = []
labels = []
original_images = []

for file in os.listdir(data_path):

    image_path = os.path.join(data_path, file)
    image = cv2.imread(image_path)

    if image is None:
        continue

    display_image = cv2.resize(image, (128, 128))
    original_images.append(
        cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
    )

    image = cv2.resize(image, (64, 64))

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = hog(
        gray_image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2)
    )

    images.append(features)

    if file.lower().startswith("cat"):
        labels.append(0)

    elif file.lower().startswith("dog"):
        labels.append(1)

X = np.array(images)
y = np.array(labels)

print("Total images:", len(X))
print("Feature size:", X.shape[1])

X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
    X,
    y,
    original_images,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training images:", len(X_train))
print("Testing images:", len(X_test))


model = SVC(kernel="rbf", C=10)

print("Training SVM model...")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print("SVM Accuracy:", accuracy)


# predictions
plt.figure(figsize=(10, 6))

for i in range(6):

    prediction = model.predict(
        X_test[i].reshape(1, -1)
    )

    if prediction[0] == 0:
        result = "Cat"
    else:
        result = "Dog"

    plt.subplot(2, 3, i + 1)
    plt.imshow(img_test[i])
    plt.title("Predicted: " + result)
    plt.axis("off")


plt.tight_layout()
plt.show()