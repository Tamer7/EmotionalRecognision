import cv2
import numpy as np
import tensorflow as tf
from time import sleep
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from tensorflow.python.keras.backend import set_session


#Loading the hardcascade and model file into a variable
face_classifier=cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
classifier = load_model('static/EmotionDetectionModel.h5')
# Labels to make predictions on
class_labels=['Angry','Happy','Neutral','Sad','Surprise']

# This class loads the video caputure and makes predictions
class DetectEmotion(object):
    def __init__(self):
        self.cap=cv2.VideoCapture(0)
    def __del__(self):
        self.cap.release()
    def get_frame(self):
        ret,frame=self.cap.read()
        labels=[]
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_classifier.detectMultiScale(gray,1.3,5)
    
        # Drawing the rectangle around the face
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),thickness=7)
            roi_gray=gray[y:y+h,x:x+w]
            roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
                
            if np.sum([roi_gray])!=0:
                roi=roi_gray.astype('float')/255.0
                roi=img_to_array(roi)
                roi=np.expand_dims(roi,axis=0)
                # Making the prediction using the .predict funtion 
                preds=classifier.predict(roi)[0]
                label=class_labels[preds.argmax()]
                label_position=(x,y)
                # Inserting the prediction into text on the screen 
                cv2.putText(frame,label,label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                cv2.putText(frame,'No Face Found',(20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return (jpeg.tobytes())



