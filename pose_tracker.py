import cv2
import mediapipe as mp
import numpy as np
from geometry_engine import GeometryEngine
from rep_tracker import RepState, calculate_next_state
from voice_engine import VoiceEngine
from ai_coach import AICoach
import dataclasses

class PoseTracker:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.85):
        self.mp_pose = mp.solutions.pose 
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode = False, 
            model_complexity=1, 
            min_detection_confidence = min_detection_confidence,
            min_tracking_confidence = min_tracking_confidence
        )

    def process_frame(self, img, draw=True):
        """returns a BGR to RGB converted image and processes the frame to save the results object in self.results"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.pose.process(img_rgb)

        if self.results.pose_landmarks and draw:
            self.mp_draw.draw_landmarks(
                img, 
                self.results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS
            )
        return img, self.results
    
    def find_positions(self, img, draw=True):
        lm_list = []
        if self.results.pose_landmarks:
            h, w, c = img.shape
        
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy, lm.z, lm.visibility])
                if draw:
                    cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)
        return lm_list
    
    def is_fully_visible(self, lm_list, threshold=0.5):
        """
        Checks if critical body parts are visible.
        Returns: True if user is in frame, False if not.
        """

        if len(lm_list) == 0: return False


        left_visible = (lm_list[11][4] > threshold) and \
                       (lm_list[23][4] > threshold) and \
                       (lm_list[25][4] > threshold)
                       

        right_visible = (lm_list[12][4] > threshold) and \
                        (lm_list[24][4] > threshold) and \
                        (lm_list[26][4] > threshold)
        

        return left_visible or right_visible

def draw_text_with_background(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, scale=1, color=(255, 255, 255), thickness=2):
    x, y = position
    line_height = 40
    
    words = text.split(' ')
    lines = []
    current_line = ""
    
    for word in words:
        if len(current_line) + len(word) > 40: 
            lines.append(current_line)
            current_line = word + " "
        else:
            current_line += word + " "
    lines.append(current_line)
    
    box_height = len(lines) * line_height + 20
    cv2.rectangle(img, (x-10, y-30), (x + 600, y + box_height), (0, 0, 0), cv2.FILLED)
    

    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y + (i * line_height)), font, scale, color, thickness)

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = PoseTracker()
    current_state = RepState()
    ai_coach = AICoach()
    voice = VoiceEngine()

    total_form_errors = 0
    worst_recorded_angle = 180

    workout_active = True
    summary_generated = False
    coach_feedback = ""

    voice.speak("System ready. Get started.")
    previous_count = 0

    while True:
        ret, frame = cap.read()

        if not ret: 
            print("can't receive frame (stream end?)...")
            break
        
        if workout_active: 
            frame, result = detector.process_frame(frame)
            lm_list = detector.find_positions(frame, draw=False)

            if (len(lm_list)) != 0:

                h, w, c = frame.shape
                is_visible = detector.is_fully_visible(lm_list)

                if not is_visible:
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                    
                    # Show Warning 
                    cv2.putText(frame, "STEP BACK", (w//4, h//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    cv2.putText(frame, "Ensure full body is in frame", (w//4 - 50, h//2 + 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            
                else:
                    def get_coords(index):
                        point = lm_list[index]
                        return [point[1], point[2], point[3] * w]
                    
                    p_shoulder = get_coords(11)
                    p_elbow = get_coords(13)
                    p_wrist = get_coords(15)

                    p_hip = get_coords(23)
                    p_knee = get_coords(25)

                    elbow_angle = GeometryEngine.calculate_angle(p_shoulder, p_elbow, p_hip)
                    hip_angle = GeometryEngine.calculate_angle(p_shoulder, p_hip, p_knee)
                    
                    current_state = calculate_next_state(current_state, elbow_angle)

                    if current_state.count == 1 and current_state.baseline_hip_angle is None:

                        current_state = dataclasses.replace(current_state, baseline_hip_angle=hip_angle)
                        voice.speak("Form calibrated.")
                        print(f"DEBUG: Baseline set to {hip_angle}")

                    if current_state.baseline_hip_angle is not None:
                        # allow 15 degrees of sagging from the baseline
                        threshold = current_state.baseline_hip_angle - 15
                    else:
                        threshold = 150 

                    if hip_angle > threshold:
                        status_color = (0, 255, 0)
                        feedback_text = "FORM: GOOD"
                    else:
                        status_color = (0, 0, 255)
                        feedback_text = "FIX HIPS!"

                    if feedback_text == "FIX HIPS!":
                        total_form_errors += 1
                    if hip_angle < worst_recorded_angle:
                        worst_recorded_angle = hip_angle

                    if current_state.count > previous_count:
                        voice.speak(str(current_state.count), cooldown=0.5)
                        previous_count = current_state.count

                    if 'feedback_text' in locals() and feedback_text == "FIX HIPS!":
                        voice.speak("Fix your hips", cooldown=3.0)    


                    cv2.line(frame, (p_shoulder[0], p_shoulder[1]), (p_elbow[0], p_elbow[1]), (0, 255, 255), 3)
                    cv2.line(frame, (p_elbow[0], p_elbow[1]), (p_wrist[0], p_wrist[1]), (0, 255, 255), 3)

                    cv2.line(frame, (p_shoulder[0], p_shoulder[1]), (p_hip[0], p_hip[1]), status_color, 3)
                    cv2.line(frame, (p_hip[0], p_hip[1]), (p_knee[0], p_knee[1]), status_color, 3)

                    cv2.circle(frame, (p_hip[0], p_hip[1]), 8, status_color, cv2.FILLED)

                    cv2.rectangle(frame, (0,0), (300, 100), (245, 117, 16), cv2.FILLED)
                    cv2.putText(frame, f"REPS: {current_state.count}", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, f"{current_state.stage}", (10, 90), 
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
                    
                    if feedback_text == "FIX HIPS!":
                        cv2.putText(frame, feedback_text, (50, 200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                        
                cv2.imshow("Frame", frame)  

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    workout_active = False # <--- Switch modes
                    voice.speak("Workout complete. Analyzing.")
    
        else:
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

            if not summary_generated:
                cv2.putText(frame, "Consulting AI Coach...", (100, 300), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                cv2.imshow("Frame", frame)
                cv2.waitKey(1)
                
                if current_state.count > 0:
                    coach_feedback = ai_coach.generate_feedback(
                        reps=current_state.count,
                        errors=total_form_errors,
                        worst_angle=worst_recorded_angle
                    )
                else:
                    coach_feedback = "No reps recorded."

                summary_generated = True
                voice.speak(coach_feedback, cooldown=0)

            cv2.putText(frame, "WORKOUT SUMMARY", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.putText(frame, f"Total Reps: {current_state.count}", (50, 160), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            cv2.putText(frame, f"Form Errors: {total_form_errors}", (50, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            
            draw_text_with_background(frame, coach_feedback, (50, 300), scale=0.8)
            
            cv2.putText(frame, "Press 'Q' again to Exit", (50, 650), cv2.FONT_HERSHEY_PLAIN, 1.5, (100, 100, 100), 2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.imshow("Frame", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


    
