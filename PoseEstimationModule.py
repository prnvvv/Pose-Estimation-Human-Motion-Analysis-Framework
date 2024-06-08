import cv2
import mediapipe as mp
import time

class PoseDetector:
    def __init__(self):
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.mpDraw = mp.solutions.drawing_utils
        self.capture = cv2.VideoCapture(0)

        if not self.capture.isOpened():
            raise Exception("Webcam is not opened.")
        
        self.currentTime = 0
        self.previousTime = 0

    def process_frame(self):
        success, image = self.capture.read()
        if not success:
            raise Exception("Failed to capture image")
        
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)
        
        if results.pose_landmarks:
            self.mpDraw.draw_landmarks(image, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
            for id, landmark in enumerate(results.pose_landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(image, (cx, cy), 10, (255, 0, 0), cv2.FILLED, 3)
        
        self.currentTime = time.time()
        fps = 1 / (self.currentTime - self.previousTime)
        self.previousTime = self.currentTime

        cv2.putText(image, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255, 0, 0), 3)

        return image

    def show_video(self):
        while True:
            try:
                image = self.process_frame()
                cv2.imshow("Image", image)
                
                if cv2.waitKey(1) == ord("q"):
                    break
            except Exception as e:
                print(e)
                break

    def release(self):
        self.capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = PoseDetector()
    try:
        detector.show_video()
    finally:
        detector.release()
