import cv2
import numpy as np #배열 계산 용이
from PIL import Image #python imaging library
import os

path = 'dataset' #경로 (dataset 폴더)
recognizer = cv2.face.LBPHFaceRecognizer_create()
'''
recognizer_E = cv2.face.EigenFaceRecognizer_create()
recognizer_F = cv2.face.FisherFaceRecognizer_create()
'''
cascadePath = 'haarcascades/haarcascade_frontalface_default.xml'
detector = cv2.CascadeClassifier(cascadePath)


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    #listdir : 해당 디렉토리 내 파일 리스트
    #path + file Name : 경로 list 만들기

    faceSamples = []
    ids = []
    for imagePath in imagePaths: #각 파일마다
        #흑백 변환
        PIL_img = Image.open(imagePath).convert('L') #L : 8 bit pixel, bw
        img_numpy = np.array(PIL_img, 'uint8')

        #user id
        id = int(os.path.split(imagePath)[-1].split(".")[1])#마지막 index : -1

        #얼굴 샘플
        faces = detector.detectMultiScale(img_numpy)
        for(x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples, ids

print('\n [INFO] Training faces. It will take a few seconds. Wait ...')
faces, ids = getImagesAndLabels(path)

recognizer.train(faces,np.array(ids)) #학습
recognizer.write('trainer/trainer.yml')

'''
recognizer_E.train(faces,np.array(ids)) 
recognizer_E.write('trainer/trainer_E.yml')
recognizer_F.train(faces,np.array(ids)) 
recognizer_F.write('trainer/trainer_F.yml')
'''
print('\n [INFO] {0} faces trained. Exiting Program'.format(len(np.unique(ids))))