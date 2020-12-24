import cv2
import time

def millis():
    return int(time.time() * 1000.0)

def stereo_read(c1, c2):
    c1.grab()
    c2.grab()
    _, img1 = c1.retrieve()
    _, img2 = c2.retrieve()
    return img1, img2

fourcc = cv2.VideoWriter_fourcc(*'XVID')

w1 = cv2.VideoWriter('./video/video1.avi', fourcc, 15.0, (1280,  720))
w2 = cv2.VideoWriter('./video/video2.avi', fourcc, 15.0, (1280,  720))

c1 = cv2.VideoCapture()
c1.open(0)
c1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)        # 720p

c2 = cv2.VideoCapture()
c2.open(1, cv2.CAP_DSHOW)
c2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)        # 720p
c2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

for i in range(30*20):
    timestamp = millis()

    img1, img2 = stereo_read(c1, c2)
    w1.write(img1)
    w2.write(img2)

    deltaTime = millis() - timestamp
    print(deltaTime)
    cv2.waitKey(max(1, 33 - deltaTime))  # adaptive loop frequency

w1.release()
w2.release()