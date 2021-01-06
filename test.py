import cv2
import numpy as np #배열 계산 용이
import time

faceCascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
'''
eyeCascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier('haarcascades/haarcascade_smile.xml')
'''

capture = cv2.VideoCapture(0) # initialize, # is camera number
capture.set(cv2.CAP_PROP_FRAME_WIDTH,1280) #CAP_PROP_FRAME_WIDTH == 3
capture.set(cv2.CAP_PROP_FRAME_HEIGHT,720) #CAP_PROP_FRAME_HEIGHT == 4
#capture.set(cv2.CAP_PROP_BRIGHTNESS,-100)

prev_time = 0
FPS = 60

while True: #영상 출력
    ret, frame = capture.read() 
    # frame = cv2.flip(frame, -1) 상하반전
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #흑백으로
    faces = faceCascade.detectMultiScale(
        gray,#검출하고자 하는 원본이미지
        scaleFactor = 1.2, #검색 윈도우 확대 비율, 1보다 커야 한다, Scale factor
        minNeighbors = 5, #얼굴 사이 최소 간격(픽셀)
        minSize=(20,20) #얼굴 최소 크기. 이것보다 작으면 무시
    )
        #### scaleFactor
        #여러 스케일에 걸쳐 분석하기 위해서 이미지 사이즈를 축소시켜 가며 만든 이미지 집합
        #이미지 사이즈를 줄일수록 스케일은 커진다
        #scaleFactor = 1.1 : 1, 1.1, 1.1*1.1 ...
        #이미지 크기는 1/scale 씩 축소됨 (https://darkpgmr.tistory.com/137)

        #### minNeighbors
        #여러 스케일 크기에서 minNeighbors 횟수 이상 검출된 object를 valid하게 검출
        #기본값 3

    #얼굴에 대해 rectangle 출력
    for (x,y,w,h) in faces:
        #(x,y) 검출된 얼굴 좌상단 위치
        # w, h 가로, 세로
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        #inputOutputArray, point1 , 2, colorBGR, thickness)
        roi_gray = gray[y:y+h, x:x+w] #Region of Interest 관심영역
        roi_color = frame[y:y+h, x:x+w]
    '''
    # 눈 검출
    eyes = eyeCascade.detectMultiScale(
        roi_gray, #얼굴 영역 안에서
        scaleFactor= 1.3,
        minNeighbors = 7, #검출하려는 크기가 작을수록 늘이는 걸까?
        minSize=(5,5)
    )
    for(ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0),2)

    # 입 검출
    smile = smileCascade.detectMultiScale(
        roi_gray,
        scaleFactor=1.3,
        minNeighbors = 20,
        minSize=(25,25)
    )
    for(sx,sy,sw,sh) in smile:
        cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0,0,255),2)
    '''
    # 화면으로 출력!
    current_time = time.time() - prev_time #경과 시간 = 현재 시간 - 이전 프레임 재생 시간
    if (ret is True) and (current_time > 1./FPS) : #일정 FPS 이상 시간 경과 시 새로운 프레임 출력
        prev_time = time.time()
        cv2.imshow("VideoFrame",frame)

    #종료 조건
        if cv2.waitKey(1) > 0 : break #cv2.waitKey(time) : time마다 키 입력상태를 받아옴, 아스키 값 반환
    #if cv2.waitKey(1) == ord('q') : break #q를 눌렀을 때
    #if (cv2.waitKey(1) & 0xff) == 27 : break #ESC 눌렀을 때
    #키 번호 참고 https://firejune.com/731/event.keyCode+%EB%B2%88%ED%98%B8%ED%91%9C 

capture.release() #메모리 해제
cv2.destroyAllWindows()#모든 윈도우 창 닫기 ; "제목"이용해서 특정 윈도우창만 닫을수도