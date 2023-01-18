# -*- coding: utf-8 -*
"""
Created on Sat Dec  5 16:49:07 2020

@author: Lenovo
"""

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# add image path and trained model path
image_path = "C:/Users/Lenovo/Desktop/college/sem7/PRML/mini_project/with_mask2.jpg"
mask_detector_model = "C:/Users/Lenovo/Desktop/college/sem7/PRML/mini_project/model.h5"

# Loading trained mask detector model
print("loading face mask detector model...")
model = load_model(mask_detector_model)

# preprocessing test image
image = load_img(image_path, target_size=(224, 224))
image = img_to_array(image)
image = preprocess_input(image)
face = np.expand_dims(image, axis=0)

# Predicting results using model
(mask, withoutMask) = model.predict(face)[0]

# Printing model predictions
label = "Mask" if mask > withoutMask else "No Mask"
label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
print(label)
