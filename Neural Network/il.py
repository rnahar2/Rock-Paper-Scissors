from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.models import Sequential, load_model
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2
import itertools
import warnings
import shutil
import random
import os.path
import glob
import os

# For categorizing images

os.chdir('imgs')
if os.path.isdir('train/rock') is False:
    os.makedirs('train/rock')
    os.makedirs('train/paper')
    os.makedirs('train/scissors')
    os.makedirs('valid/rock')
    os.makedirs('valid/paper')
    os.makedirs('valid/scissors')
    os.makedirs('test/rock')
    os.makedirs('test/paper')
    os.makedirs('test/scissors')

    for c in random.sample(glob.glob('rock*'), 100):
        shutil.move(c, 'train/rock')
    for c in random.sample(glob.glob('paper*'), 100):
        shutil.move(c, 'train/paper')
    for c in random.sample(glob.glob('scissors*'), 100):
        shutil.move(c, 'train/scissors')
    for c in random.sample(glob.glob('rock*'), 25):
        shutil.move(c, 'valid/rock')
    for c in random.sample(glob.glob('paper*'), 25):
        shutil.move(c, 'valid/paper')
    for c in random.sample(glob.glob('scissors*'), 25):
        shutil.move(c, 'valid/scissors')
    for c in random.sample(glob.glob('rock*'), 10):
        shutil.move(c, 'test/rock')
    for c in random.sample(glob.glob('paper*'), 10):
        shutil.move(c, 'test/paper')
    for c in random.sample(glob.glob('scissors*'), 10):
        shutil.move(c, 'test/scissors')

os.chdir('C:\LSU\Spring 2022\EE 4780\Leanring NN')

train_path = 'imgs/train'
valid_path = 'imgs/valid'
test_path = 'imgs/test'

#For preprocessing to get ready for the NN

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224, 224), classes=['rock', 'paper', 'scissors'],
                         batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224, 224), classes=['rock', 'paper', 'scissors'],
                         batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224, 224), classes=['rock', 'paper', 'scissors'],
                         batch_size=6, shuffle=False)

#To make sure everything saved correctly


assert train_batches.n == 300
assert valid_batches.n == 75
assert test_batches.n == 30
assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == 3

#imgs, labels = next(train_batches)


'''def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()'''

#The Sequential model

model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=3, activation='softmax'),
])

'''To check setup
model.summary()
'''

#Running the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_batches, validation_data=valid_batches, epochs=20, steps_per_epoch=5)

#For saving to the model to call and insert input
if os.path.isfile('models/3UC.h5') is False:
    model.save('models/3UC.h5')

#new_model = load_model('models/3UC.h5')

'''while True:
    data = input()
    if str.lower(data) == "q":
        break
    NNin.append(data)'''

'''#How we will pass our input
nnin = cv2.imread('imgs/paper.jpg')
imgResize = cv2.resize(nnin, (224, 224))
imgResize = imgResize[np.newaxis, :]
print(imgResize.shape)

predictions = model.predict(x=imgResize, steps=1, verbose=0)
print(np.argmax(predictions))'''

