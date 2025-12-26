import cv2
import mediapipe as mp
import numpy as np

class PoseTracker:
    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        #initializing MediaPipe pose
        self.mp_pose = mp.solutions.pose 
        self.mp_draw = mp.solutions.drawing_utils
        # calling the Pose class from mediapipe.solutions.pose and creating an object self.pose with it
        self.pose = self.mp_pose.Pose(
            static_image_mode = False, #setting it to false makes the model treat images as a video stream.
            model_complexity=1, # 0=Lite, 1=Full, 2=Heavy
            min_detection_confidence = min_detection_confidence,
            min_tracking_confidence = min_tracking_confidence
        )

    def find_pose(self, img, draw=True):
        """converts BGR to RGB and processes the frame"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.pose.process(img_rgb)

        if self.results.pose_landmarks and draw:
            self.mp_draw.draw_landmarks(
                img, 
                self.results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS #this tells which dots to connect. (e.g. left shoulder to right shoulder)
            )
        return img, self.results
    
    def find_positions(self, img, draw=True):
        lm_list = []
        if self.results.pose_landmarks:
            h, w, c = img.shape

            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy, lm.z])
                if draw:
                    cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)
        return lm_list
    
def main():
    cap = cv2.VideoCapture(0)
    detector = PoseTracker()

    while True:
        ret, frame = cap.read()
        frame, result = detector.find_pose(frame)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
# if run directly, main is executed. python internally sets __name__ = __main__ but if I'm importing it __name__ == __main__ is false
if __name__ == "__main__":
    main()


    