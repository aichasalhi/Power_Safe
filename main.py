import keras as k
import cv2 as cv
import numpy as np
import time
import sys


Emotions = ['En colère', 'Dégoûter',
            'Peur', 'heureux(se)', 'Neutre', 'Triste']
capture = cv.VideoCapture(0)
timeout = time.time() + 60*1


faces_detector = cv.CascadeClassifier(
    r'/home/imad/Desktop/Devfest/Emotion_Detection_CNN-main/haarcascade_frontalface_default.xml')
model = k.models.load_model(
    r'/home/imad/Desktop/Devfest/Emotion_Detection_CNN-main/model.h5')

while True:
    _, frame = capture.read()
    calculated_emotion = [0, 0, 0, 0, 0, 0]
    img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = faces_detector.detectMultiScale(img)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        detection_face = img[y:y+h, x:x+w]
        detection_face = cv.resize(detection_face, (48, 48),
                                   interpolation=cv.INTER_AREA)

        if np.sum([detection_face]) != 0:
            emotion = detection_face.astype('float')/255.0
            emotion = k.preprocessing.image.img_to_array(emotion)
            emotion = np.expand_dims(emotion, axis=0)

            prediction = model.predict(emotion)[0]
            for i in range(0, 6):
                calculated_emotion[i] = calculated_emotion[i] + prediction[i]
            if time.time() > timeout:
                print(Emotions[np.argmax(calculated_emotion)])
                timeout = time.time() + 60*1
                calculated_emotion = [0, 0, 0, 0, 0, 0]

            live_emotion = Emotions[prediction.argmax()]
            cv.putText(frame, live_emotion, (x, y),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv.putText(frame, 'Pas de visages', (30, 80),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.imshow("Détecteur d'émotions", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv.destroyAllWindows()
