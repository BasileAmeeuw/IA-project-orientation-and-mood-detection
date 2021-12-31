import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array 
from keras.models import  load_model, model_from_json
import matplotlib.pyplot as plt
import tensorflow as tf

with open("models/model.json", "r") as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("models/model_weights.h5")
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

def predict_emotion(img):
    preds = loaded_model.predict(img)
    return EMOTIONS_LIST[np.argmax(preds)]

EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

cap = cv2.VideoCapture(0)

while True:
    _,img=cap.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_img, 1.2, 5)
    print(faces)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi = gray_img[y:y + w, x:x + h]
        roi = cv2.resize(roi, (48, 48))
        pred = loaded_model.predict(roi[np.newaxis, :, :, np.newaxis])
        max_index = np.argmax(pred[0])
        
        emotions = ('angry','disgust','fear','happy','neutral','sad','surprise')
        predicted_emotion = emotions[max_index]
        print(predicted_emotion)
        cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.putText(img, pred, (x, y), font, 1, (255, 255, 0), 2)
    resized_img = cv2.resize(img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows

    # for (x, y, w, h) in faces:
    #     fc = gray_img[y:y+h, x:x+w]

    #     roi = cv2.resize(fc, (48, 48))
    #     pred = loaded_model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

    #     cv2.putText(img, pred, (x, y), font, 1, (255, 255, 0), 2)
    #     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    # resized_img = cv2.resize(img, (1000, 700))
    # cv2.imshow('Facial emotion analysis ', resized_img)