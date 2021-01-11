import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
trainerPath = 'trainer/trainer.yml'
recognizer.read(trainerPath)
cascadePath = 'haarcascades/haarcascade_frontalface_alt2.xml' #'haarcascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascadePath)

'''
recognizer_E = cv2.face.EigenFaceRecognizer_create()
recognizer_E.read('trainer/trainer_E.yml')
recognizer_F = cv2.face.FisherFaceRecognizer_create()
recognizer_F.read('trainer/trainer_F.yml')
'''
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0

names = ['None','sumin','dongjun','minji','umma','eunho']

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1980)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

minW = 0.1 * cam.get(cv2.CAP_PROP_FRAME_WIDTH)
minH = 0.1 * cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=7,
        minSize=(int(minW), int(minH))
    )

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),2)
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        
        #id_E = recognizer_E.predict(gray[y:y+h, x:x+w])
        #id_F = recognizer_E.predict(gray[y:y+h, x:x+w])

        #if id == id_E == id_F :
        if confidence < 55 :
            id = names[id]
        else:
            id = "unknown"
            
        confidence = "  {0}%".format(round(100-confidence))

        cv2.putText(img,str(id), (x+5,y-5),font,1,(255,255,255),2)
        cv2.putText(img,str(confidence), (x+5,y+h-5),font,1,(255,255,0),1)
    
    cv2.imshow('camera',img)
    if cv2.waitKey(1) > 0 : break

print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()