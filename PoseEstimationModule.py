import cv2
import mediapipe as mp
import time

class PoseDetector:
    def __init__(self, static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            exit()

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mpDraw = mp.solutions.drawing_utils

    def detect_pose(self):
        previousTime = 0
        currentTime = 0

        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("Error: Could not read frame from webcam.")
                break

            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(imgRGB)

            if results.pose_landmarks:
                self.mpDraw.draw_landmarks(frame, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
                for id, landmark in enumerate(results.pose_landmarks.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    if id in [0, 11, 12, 23, 24]:  # Highlighting specific landmarks
                        cv2.circle(frame, (cx, cy), 10, (255, 0, 0), cv2.FILLED, 3)

            currentTime = time.time()
            fps = 1 / (currentTime - previousTime)
            previousTime = currentTime

            cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 0, 0), 3)
            cv2.imshow("Webcam Frame", frame)

            if cv2.waitKey(1) == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    pose_detector = PoseDetector(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    pose_detector.detect_pose()
