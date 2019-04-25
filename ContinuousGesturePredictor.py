import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import numpy as np
from PIL import Image
import cv2
import imutils
import urllib.request
from directkeys import PressKey, W, A, S, D, K, ReleaseKey

bg = None
url='http://192.168.137.68:8080/shot.jpg'

def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(imageName)

def run_avg(image, aWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return

    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)

    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    (_, cnts, _) = cv2.findContours(thresholded.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

def main():
    aWeight = 0.5

    #camera = cv2.VideoCapture(0)

    top, right, bottom, left = 10, 350, 225, 590

    num_frames = 0
    start_recording = False

    while(True):
        #(grabbed, frame) = camera.read()
        imgResp = urllib.request.urlopen(url)
        imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
        frame = cv2.imdecode(imgNp,-1)
        frame = imutils.resize(frame, width = 700)

        frame = cv2.flip(frame, 1)

        clone = frame.copy()

        (height, width) = frame.shape[:2]

        roi = frame[top:bottom, right:left]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            hand = segment(gray)

            if hand is not None:
                (thresholded, segmented) = hand
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                if start_recording:
                    cv2.imwrite('Temp.png', thresholded)
                    resizeImage('Temp.png')
                    predictedClass, confidence = getPredictedClass()
                    showStatistics(predictedClass, confidence)
                cv2.imshow("Thesholded", thresholded)

        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        num_frames += 1

        cv2.imshow("Video Feed", clone)

        keypress = cv2.waitKey(1) & 0xFF

        if keypress == ord("q"):
            break
        
        if keypress == ord("s"):
            start_recording = True

def getPredictedClass():
    image = cv2.imread('Temp.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    prediction = model.predict([gray_image.reshape(89, 100, 1)])
    return np.argmax(prediction), (np.amax(prediction) / (prediction[0][0] + prediction[0][1] + prediction[0][2] + prediction[0][3] + prediction[0][4]))

def showStatistics(predictedClass, confidence):

    textImage = np.zeros((300,512,3), np.uint8)
    className = ""

    if predictedClass == 0:
        className = "Swing"
        '''ReleaseKey(A)
        ReleaseKey(W)
        ReleaseKey(S)
        PressKey(D)'''
    elif predictedClass == 1:
        className = "Palm"
        '''ReleaseKey(A)
        ReleaseKey(D)
        ReleaseKey(S)
        PressKey(W)'''
    elif predictedClass == 2:
        className = "Fist"
        '''ReleaseKey(W)
        PressKey(S)
        ReleaseKey(D)
        ReleaseKey(A)'''
    elif predictedClass == 3:
        className = "Victory"
    elif predictedClass == 4:
        className = "Index"

    cv2.putText(textImage,"Pedicted Class : " + className, 
    (30, 30), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    1,
    (255, 255, 255),
    2)

    cv2.putText(textImage,"Confidence : " + str(confidence * 100) + '%', 
    (30, 100), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    1,
    (255, 255, 255),
    2)
    cv2.imshow("Statistics", textImage)




# Model defined
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

# Load Saved Model
model.load("Trained_Models/GRM5gesture.tfl")

main()