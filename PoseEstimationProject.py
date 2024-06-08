import cv2
import mediapipe as mp
import time

mpPose = mp.solutions.pose
Pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

capture = cv2.VideoCapture(0)

if capture.isOpened == 0:
    print("Webcam is not opened.")
    exit()

currentTime = 0
previousTime = 0

while True:
    success, image = capture.read()
    
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = Pose.process(imgRGB)
    
    landmarks = results.pose_landmarks
    if landmarks:
        mpDraw.draw_landmarks(image, landmarks, mpPose.POSE_CONNECTIONS)
        for id, landmark in enumerate(landmarks):
            h, w, c = image.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (cx, cy), 10, (255, 0, 0), cv2.FILLED, 3)
    currentTime = time.time()
    fps = 1 /( currentTime - previousTime )
    previousTime = currentTime

    cv2.putText(image, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255, 0, 0), 3)

    cv2.imshow("Image",image)
    
    if cv2.waitKey(1) == ord("q"):
        break

capture.release
cv2.destroyAllWindows