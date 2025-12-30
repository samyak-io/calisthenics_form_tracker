import cv2
import mediapipe as mp
import numpy as np
from geometry_engine import GeometryEngine
from rep_tracker import RepState, calculate_next_state

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
    current_state = RepState()

    while True:
        ret, frame = cap.read()

        if not ret: 
            break
        
        
        frame, result = detector.find_pose(frame)
        lm_list = detector.find_positions(frame, draw=False)

        if (len(lm_list)) != 0:
            h, w, c = frame.shape

            # MediaPipe IDs: 11=Left Shoulder, 13=Left Elbow, 15=Left Wrist
            def get_coords(index):
                point = lm_list[index]
                return [point[1], point[2], point[3] * w]
            
            p1 = get_coords(11)
            p2 = get_coords(13)
            p3 = get_coords(15)

            angle = GeometryEngine.calculate_angle(p1, p2, p3)

            # brain of the tracker
            current_state = calculate_next_state(current_state, angle)

            # draw the lines and angle at elbow 
            # syntax:
            # cv2.putText(image, text, org, font, fontScale, color, thickness=1, lineType=cv2.LINE_8, bottomLeftOrigin=False)
            cv2.putText(frame, str(int(angle)), (p2[0] - 20, p2[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            # draw the rep counter box
            # image, start point, end point, color, thickness
            cv2.rectangle(frame, (0,0), (250, 100), (245,117, 16), cv2.FILLED)
            cv2.putText(frame, f"REPS: {current_state.count}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(frame, f"STAGE: {current_state.stage}", (10, 90), 
                       cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

        cv2.imshow("Frame", frame)  
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# if run directly, main is executed. python internally sets __name__ = __main__ but if I'm importing it __name__ == __main__ is false
if __name__ == "__main__":
    main()


    