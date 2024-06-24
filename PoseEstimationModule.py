import cv2  # OpenCV library for computer vision tasks
import mediapipe as mp  # MediaPipe library for pose detection
import time  # Time library to calculate frame rate

class PoseDetector:
    def __init__(self, static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Initialize video capture object to read from webcam
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            # Print error message if webcam cannot be opened
            print("Error: Could not open webcam.")
            exit()

        # Initialize MediaPipe pose solution
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        # Initialize MediaPipe drawing utility
        self.mpDraw = mp.solutions.drawing_utils

    def detect_pose(self):
        # Variables to calculate frames per second (FPS)
        previousTime = 0
        currentTime = 0

        while True:
            # Read frame from webcam
            ret, frame = self.cap.read()

            if not ret:
                # Print error message if frame cannot be read
                print("Error: Could not read frame from webcam.")
                break

            # Convert the frame to RGB as MediaPipe uses RGB format
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Process the frame to detect pose
            results = self.pose.process(imgRGB)

            if results.pose_landmarks:
                # Draw pose landmarks on the frame
                self.mpDraw.draw_landmarks(frame, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
                for id, landmark in enumerate(results.pose_landmarks.landmark):
                    # Get dimensions of the frame
                    h, w, c = frame.shape
                    # Calculate coordinates of the landmark
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    if id in [0, 11, 12, 23, 24]:  # Highlighting specific landmarks
                        # Draw circles on specific landmarks
                        cv2.circle(frame, (cx, cy), 10, (255, 0, 0), cv2.FILLED, 3)

            # Calculate FPS
            currentTime = time.time()
            fps = 1 / (currentTime - previousTime)
            previousTime = currentTime

            # Display FPS on the frame
            cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 0, 0), 3)
            # Show the frame with pose landmarks
            cv2.imshow("Webcam Frame", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) == ord('q'):
                break

        # Release the webcam and close all OpenCV windows
        self.cap.release()
        cv2.destroyAllWindows()

# Main function to create a PoseDetector object and start pose detection
if __name__ == "__main__":
    pose_detector = PoseDetector()
    pose_detector.detect_pose()
