import cv2
import numpy as np #배열 계산 용이
import os

faceCascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

capture = cv2.VideoCapture(0) # initialize, # is camera number
capture.set(cv2.CAP_PROP_FRAME_WIDTH,1280) #CAP_PROP_FRAME_WIDTH == 3
capture.set(cv2.CAP_PROP_FRAME_HEIGHT,720) #CAP_PROP_FRAME_HEIGHT == 4

face_id = input('\n enter user id end press <return> ==> ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")

count = 0

while True: #영상 출력
    ret, frame = capture.read() 
    # frame = cv2.flip(frame, -1) 상하반전
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #흑백으로
    faces = faceCascade.detectMultiScale(
        gray,#검출하고자 하는 원본이미지
        scaleFactor = 1.2, #검색 윈도우 확대 비율, 1보다 커야 한다, Scale factor
        minNeighbors = 6, #얼굴 사이 최소 간격(픽셀)
        minSize=(20,20) #얼굴 최소 크기. 이것보다 작으면 무시
    )

    #얼굴에 대해 rectangle 출력
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        count += 1
        cv2.imwrite("dataset/User."+str(face_id)+'.'+str(count)+".jpg",gray[y:y+h, x:x+w])
        cv2.imshow('image',frame)

    if cv2.waitKey(1) > 0 : break
    elif count >= 50 : break #30 face sample

print("\n [INFO] Exiting Program and cleanup stuff")

capture.release() #메모리 해제
cv2.destroyAllWindows()#모든 윈도우 창 닫기 ; "제목"이용해서 특정 윈도우창만 닫을수도
