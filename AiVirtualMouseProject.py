import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

wCam = 640
hCam = 480
frame_reduce = 100
smoothness = 7

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
# pTime = 0
detector = htm.handDetector(maxHands=1)
wSrc, hSrc = autopy.screen.size()
# print(wSrc, hSrc)

while True:
    success, img = cap.read()
    # img Manipulate
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    # Tip Position of Index Finger and Middle:
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # print(x1, y1, x2, y2)
        # Check Which Finger is Up:
        fingers = detector.fingersUp()
        # print(fingers)

        # Frame Reduce :
        cv2.rectangle(img, (frame_reduce, frame_reduce), (wCam - frame_reduce, hCam - frame_reduce),
                      (255, 0, 255), 2)


        # Only Index Finger Moving Mode:
        if fingers[1] == 1 and fingers[2] == 0:

            # Convert Coordinates:
            x3 = np.interp(x1, (frame_reduce, wCam - frame_reduce), (0, wSrc))
            y3 = np.interp(y1, (frame_reduce, hCam - frame_reduce), (0, hSrc))

            # Smoothness Values
            clocX = plocX + (x3 - plocX) / smoothness
            clocY = plocY + (y3 - plocY) / smoothness
            # & Moving mouse :
            autopy.mouse.move(wSrc - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        if fingers[1] == 1 and fingers[2] == 1:
            # Find Distance between fingers:
            length, img, line_info = detector.findDistance(8, 12, img)
            print(length)

            # Click Mouse When distance is short:
            if length < 40:
                cv2.circle(img, (line_info[4], line_info[5]), 15,
                           (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 3,
                (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
