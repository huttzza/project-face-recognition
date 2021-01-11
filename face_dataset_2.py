# cf. https://gist.github.com/ageitgey/82d0ea0fdb56dc93cb9b716e7ceb364b
# https://github.com/kairess/face_detector/blob/master/main.py

import dlib
import cv2
import sys
import numpy as np
import openface

predictor_model = "shape_predictor_68_face_landmarks.dat"

face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)

face_aligner = openface.AlignDlib(predictor_model) #https://m.blog.naver.com/kjh3864/221219659663

scaler = 0.3

face_id = input('\n enter user id # and press <return> ==> ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1980)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#face_roi = []

count = 0

while True :
    ret, img = cam.read()
    if not ret :
        break
    
    '''
    if len(face_roi) == 0 :
        faces = face_detector(img, 1)
    else :
        roi_img = img[face_roi[0]:face_roi[1], face_roi[2]:face_roi[3]]
        faces = face_detector(roi_img)
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)

    '''
    for face in faces :
        if len(face_roi) == 0:
            dlib_shape = face_pose_predictor(img, face)
            shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])
        else :
            dlib_shape = face_pose_predictor(roi_img, face)
            shape_2d = np.array([[p.x + face_roi[2], p.y+face_roi[0]] for p in dlib_shape.parts()])

        for s in shape_2d :
            cv2.circle(img, center=tuple(s), radius=1,color=(255,255,255),thickness=2)
    '''
    for face in faces: #https://m.blog.naver.com/PostView.nhn?blogId=zzing0907&logNo=221612308385&proxyReferer=https:%2F%2Fwww.google.com%2F
        l = face.left()
        t = face.top()
        b = face.bottom()
        r = face.right()
        cv2.rectangle(img,(l,t),(r,b),(255,255,255),2)
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
        #count += 1
        #cv2.imwrite("dataset/User."+str(face_id)+'.'+str(count)+".jpg",gray[y:y+h, x:x+w])

        #crop_img = img_full[d.top():d.bottom(),d.left():d.right()]
        #cv2.imwrite("cropped.jpg", crop)
        count+=1
        #(imgDim, rgbImg, bb(face), )
        # imgDim=534 : 534*534 이미지로 반환 
        # bb dlib.rectangle
        # landmarkIndices : 변환 대상 인덱스
        aligned_face = face_aligner.align(534, gray, face, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        cv2.imwrite("dataset/User."+str(face_id)+'.'+str(count)+".jpg",aligned_face)

    cv2.imshow('face',img)
    if cv2.waitKey(1) == ord('q') or count >= 30:
        break

cam.release()
cv2.destroyAllWindows()
    