import tensorflow as tf
import numpy as np
import cv2

# Load trained model
model = tf.keras.models.load_model("banana_model.h5")

# Class labels
classes = ["overripe", "ripe", "unripe"]

# Input image
img_path = input("Enter banana image path: ")
img = cv2.imread(img_path)

if img is None:
    print("Image not found")
    exit()

img = cv2.resize(img, (100,100))
img = img / 255.0
img = np.reshape(img, (1,100,100,3))

# Prediction
prediction = model.predict(img)
result = classes[np.argmax(prediction)]

print("Banana Ripeness:", result)
