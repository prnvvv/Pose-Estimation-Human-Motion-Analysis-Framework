import cv2  # OpenCV library for computer vision tasks
import mediapipe as mp  # MediaPipe library for pose detection
import time  # Time library to calculate frame rate

# Initialize MediaPipe pose solution
mpPose = mp.solutions.pose
Pose = mpPose.Pose()
# Initialize MediaPipe drawing utility
mpDraw = mp.solutions.drawing_utils

# Initialize video capture object to read from webcam
capture = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if capture.isOpened() == 0:
    print("Webcam is not opened.")
    exit()

# Variables to calculate frames per second (FPS)
currentTime = 0
previousTime = 0

while True:
    # Read frame from webcam
    success, image = capture.read()

    # Convert the frame to RGB as MediaPipe uses RGB format
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the frame to detect pose
    results = Pose.process(imgRGB)
    
    # Get the pose landmarks
    landmarks = results.pose_landmarks
    if landmarks:
        # Draw pose landmarks on the frame
        mpDraw.draw_landmarks(image, landmarks, mpPose.POSE_CONNECTIONS)
        for id, landmark in enumerate(landmarks.landmark):
            # Get dimensions of the frame
            h, w, c = image.shape
            # Calculate coordinates of the landmark
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            # Draw circles on the landmarks
            cv2.circle(image, (cx, cy), 10, (255, 0, 0), cv2.FILLED, 3)

    # Calculate FPS
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    # Display FPS on the frame
    cv2.putText(image, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255, 0, 0), 3)

    # Show the frame with pose landmarks
    cv2.imshow("Image", image)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release the webcam and close all OpenCV windows
capture.release()
cv2.destroyAllWindows()
