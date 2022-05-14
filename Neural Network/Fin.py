import cv2
import tensorflow as tf
import numpy as np

nnin = cv2.imread('imgs/rock01-000.png')        # Using opencv to save image variable as tensor
imgResize = cv2.resize(nnin, (224, 224))        # Resizing tensor to acceptable input size
imgResize = imgResize[np.newaxis, :]            # Resizing tensor to acceptable input dimension

model = tf.keras.models.load_model("P.model")   # Loads te full model

# For entering single image into model. Where x is a tensor, steps is the number of batches
# and verbose is used to give info on model. 0=silent
prediction = model.predict(x=imgResize, steps=1, verbose=0)

# To print the prediction as number (0=rock, 1=paper, 2=scissors) for feeding to main code.
print('prediction: ', np.argmax(prediction))

