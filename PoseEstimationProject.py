import cv2
import mediapipe as mp
import time

capture = cv2.VideoCapture(0)

if capture.isOpened == 0:
    print("Webcam is not opened.")
    exit()

currentTime = 0
previousTime = 0

while True:
    success, image = capture.read()
    
    currentTime = time.time()
    fps = 1 /( currentTime - previousTime )
    previousTime = currentTime

    cv2.putText(image, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255, 0, 0), 3)

    cv2.imshow("Image",image)
    
    if cv2.waitKey(1) == ord("q"):
        break

capture.release
cv2.destroyAllWindows