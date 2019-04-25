import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import numpy as np
import cv2
from sklearn.utils import shuffle

#Load Images from Swing
loadedImages = []
for i in range(0, 1000):
    image = cv2.imread('Dataset/SwingImages/swing_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))

#Load Images From Palm
for i in range(0, 1000):
    image = cv2.imread('Dataset/PalmImages/palm_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))
    
#Load Images From Fist
for i in range(0, 1000):
    image = cv2.imread('Dataset/FistImages/fist_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))
    
#Load Images From Victory
for i in range(0, 1000):
    image = cv2.imread('Dataset/VictoryImages/victory_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))
    
#Load Images From Index
for i in range(0, 1000):
    image = cv2.imread('Dataset/IndexImages/index_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))

# Create OutputVector

outputVectors = []
for i in range(0, 1000):
    outputVectors.append([1, 0, 0, 0, 0])

for i in range(0, 1000):
    outputVectors.append([0, 1, 0, 0, 0])

for i in range(0, 1000):
    outputVectors.append([0, 0, 1, 0, 0])
    
for i in range(0, 1000):
    outputVectors.append([0, 0, 0, 1, 0])
    
for i in range(0, 1000):
    outputVectors.append([0, 0, 0, 0, 1])

testImages = []

#Load Images for swing
for i in range(0, 100):
    image = cv2.imread('Dataset/SwingTest/swing_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))

#Load Images for Palm
for i in range(0, 100):
    image = cv2.imread('Dataset/PalmTest/palm_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))
    
#Load Images for Fist
for i in range(0, 100):
    image = cv2.imread('Dataset/FistTest/fist_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))
    
#Load Images for Victory
for i in range(0, 100):
    image = cv2.imread('Dataset/VictoryTest/victory_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))
    
#Load Images From Index
for i in range(0, 100):
    image = cv2.imread('Dataset/IndexTest/index_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))

testLabels = []

for i in range(0, 100):
    testLabels.append([1, 0, 0, 0, 0])
    
for i in range(0, 100):
    testLabels.append([0, 1, 0, 0, 0])

for i in range(0, 100):
    testLabels.append([0, 0, 1, 0, 0])
    
for i in range(0, 100):
    testLabels.append([0, 0, 0, 1, 0])
    
for i in range(0, 100):
    testLabels.append([0, 0, 0, 0, 1])

# Define the CNN Model
tf.reset_default_graph()
convnet=input_data(shape=[None,89,100,1],name='input')
convnet=conv_2d(convnet,32,2,activation='relu')
convnet=max_pool_2d(convnet,2)
convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,256,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,256,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=fully_connected(convnet,1000,activation='relu')
convnet=dropout(convnet,0.75)

convnet=fully_connected(convnet,5,activation='softmax')

convnet=regression(convnet,optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy',name='regression')

model=tflearn.DNN(convnet,tensorboard_verbose=0)

# Shuffle Training Data
loadedImages, outputVectors = shuffle(loadedImages, outputVectors, random_state=0)

# Train model
model.fit(loadedImages, outputVectors, n_epoch=50,
           validation_set = (testImages, testLabels),
           snapshot_step=100, show_metric=True, run_id='minor')

model.save("GRM5gesture.tfl")

