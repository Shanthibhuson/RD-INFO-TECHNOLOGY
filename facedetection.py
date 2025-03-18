import cv2
import os
import numpy as np


if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Initialize the face detector and recognizer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()


faces = []
labels = []
label_dict = {}

# Capture images for training (press 'ESC' to stop)
cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces_detected:
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        
        face_image = gray[y:y + h, x:x + w]
        
        
        faces.append(face_image)
        labels.append(count)

        
        cv2.imwrite(f"dataset/User.{count}.{str(len(faces))}.jpg", face_image)
        
        count += 1
    
    
    cv2.imshow('Training Faces', frame)

    
    if cv2.waitKey(1) & 0xFF == 27:
        break


recognizer.train(faces, np.array(labels))


recognizer.save('trainer.yml')

cap.release()
cv2.destroyAllWindows()
