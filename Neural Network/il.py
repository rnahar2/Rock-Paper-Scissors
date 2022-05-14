from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import shutil
import random
import glob
import os

# The following program is designed to take in image data and filter through it to separate
# it into the class we would like our model to predict (rock, paper, scissors).
# This data is then preprocessed and batched to be feed into a neural network model
# where is then trained, validated, and tested to predict either of the 3 outputs of
# rock, paper, or scissors.

os.chdir('imgs')    # Here we use the os package to change the directory from the current parent
                    # directory to the folder imgs where all the images that are to be used for
                    # training, validation, and testing are stored.
if os.path.isdir('train/rock') is False:    # If folders are not already created for separating our
    os.makedirs('train/rock')               # images into train, validation, and test categories, we
    os.makedirs('train/paper')              # are creating them with their respective classifications
    os.makedirs('train/scissors')
    os.makedirs('valid/rock')
    os.makedirs('valid/paper')
    os.makedirs('valid/scissors')
    os.makedirs('test/rock')
    os.makedirs('test/paper')
    os.makedirs('test/scissors')

    for c in random.sample(glob.glob('rock*'), 100):        # Here we are using the glob package to
        shutil.move(c, 'train/rock')                        # feed random image into their respective
    for c in random.sample(glob.glob('paper*'), 100):       # directories based on the names they are
        shutil.move(c, 'train/paper')                       # saved as. Images of rock shape must
    for c in random.sample(glob.glob('scissors*'), 100):    # begin with rock, etc. We feed 100 images
        shutil.move(c, 'train/scissors')                    # to each training type, 25 to each
    for c in random.sample(glob.glob('rock*'), 25):         # validation type, and 10 to each test
        shutil.move(c, 'valid/rock')                        # types
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

os.chdir('C:\LSU\Spring 2022\EE 4780\Leanring NN')          # Returns us to our previous directory

train_path = 'imgs/train'       # Set paths to variables
valid_path = 'imgs/valid'
test_path = 'imgs/test'

# Using the Keras package Image Data Generator we preprocess our image dta into tensors with the
# directories we are feeding from, the size we are shaping the images to, classes, and the size
# of the batches. The Image Data Generator returns a tensor.

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224, 224), classes=['rock', 'paper', 'scissors'],
                         batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224, 224), classes=['rock', 'paper', 'scissors'],
                         batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224, 224), classes=['rock', 'paper', 'scissors'],
                         batch_size=6, shuffle=False)

# The following helps validate that the batches indeed has the amount of images we intended to pass
# into the batches and the number of classes they are separated into.
# Where classes*ImagesPerClass=TotalBatchSize

assert train_batches.n == 300
assert valid_batches.n == 75
assert test_batches.n == 30
assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == 3

# We are using a Keras Sequential Model that has 2 hidden convolutional layers. Where the first
# hidden layer has 32 filters, a kernal size of 3x3, activation using the rectified linear, no
# padding, and accepts a tensor of rows=224, cols=224, and channels=3. We also downsample input,
# using the 2x2 pool size and strides of 2, to half. The second hidden layer is the same just
# with double the filters. Data is then flattened and we output to 3 neurons with softmax activation.

model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=3, activation='softmax'),
])

# Used to validate correct model setup
# model.summary()


# Here we compile the model using the Adam optimizer with a learning rate of 0.0001 and so it calculates
# loss and accuracy. After the model is compiled we then fit the data from the batch variables and set
# the epochs and steps accordingly
#
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_batches, validation_data=valid_batches, epochs=20, steps_per_epoch=10)

# Insert result data here

# For saving to the model to call for prediction
model.save('P.model')


# For manually passing an image into the NN. More info in main file.
'''nnin = cv2.imread('imgs/rock.jpg')
imgResize = cv2.resize(nnin, (224, 224))
imgResize = imgResize[np.newaxis, :]

predictions = model.predict(x=imgResize, steps=1, verbose=0)
print(np.argmax(predictions))'''
