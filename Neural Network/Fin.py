import cv2
import tensorflow as tf
import numpy as np
from il import test_batches

nnin = cv2.imread('imgs/rock6.jpg')
imgResize = cv2.resize(nnin, (224, 224))
imgResize = imgResize[np.newaxis, :]
#print(imgResize.shape)

model = tf.keras.models.load_model("models/3UC.h5")

prediction = model.predict(x=imgResize, steps=1, verbose=0)
print(np.argmax(prediction))

