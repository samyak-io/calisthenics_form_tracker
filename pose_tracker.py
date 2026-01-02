import cv2
import mediapipe as mp
import numpy as np
from geometry_engine import GeometryEngine
from rep_tracker import RepState, calculate_next_state
from voice_engine import VoiceEngine
from ai_coach import AICoach
import time

def draw_text_with_background(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, scale=1, color=(255, 255, 255), thickness=2):
    """
    Draws multi-line text with a dark background for readability.
    """
    x, y = position
    line_height = 40
    
    # split the text into lines
    words = text.split(' ')
    lines = []
    current_line = ""
    
    for word in words:
        if len(current_line) + len(word) > 40: # 40 chars per line width limit
            lines.append(current_line)
            current_line = word + " "
        else:
            current_line += word + " "
    lines.append(current_line)
    
    # draw background nox
    box_height = len(lines) * line_height + 20
    cv2.rectangle(img, (x-10, y-30), (x + 600, y + box_height), (0, 0, 0), cv2.FILLED)
    
    # draw text
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y + (i * line_height)), font, scale, color, thickness)

class PoseTracker:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.85):
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
                lm_list.append([id, cx, cy, lm.z, lm.visibility])
                if draw:
                    cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)
        return lm_list
    
    def is_fully_visible(self, lm_list, threshold=0.5):
        """
        Checks if critical body parts are visible.
        Returns: True if user is in frame, False if not.
        """
        # MediaPipe IDs:
        # 11=Left Shoulder, 12=Right Shoulder
        # 23=Left Hip, 24=Right Hip
        # 15=Left Wrist, 16=Right Wrist
        # 25=Left Knee, 26=Right Knee
        
        # i need to check if these points exist in our list AND have good visibility
        # Note: lm_list structure is [id, x, y, z, visibility] 
        pass

def main():
    # ... setup ...
    cap = cv2.VideoCapture(0)

    # force high resolution for better text reading
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = PoseTracker()
    current_state = RepState()
    ai_coach = AICoach()
    voice = VoiceEngine()

    #stats
    total_form_errors = 0
    worst_recorded_angle = 180

    # flags
    workout_active = True
    summary_generated = False
    coach_feedback = ""

    voice.speak("System ready. Get started.")
    previous_count = 0

    while True:
        ret, frame = cap.read()

        if not ret: break
        
        if workout_active: 
            # === PART 1: THE WORKOUT ===
            frame, result = detector.find_pose(frame)
            lm_list = detector.find_positions(frame, draw=False)

            if (len(lm_list)) != 0:

                # === angle logic ===
                h, w, c = frame.shape
                shoulder_vis = lm_list[11][4]
                hip_vis      = lm_list[23][4]
                knee_vis     = lm_list[25][4]
                
                # Threshold: 0.5 means "50% sure it's there"
                is_visible = (shoulder_vis > 0.5) and (hip_vis > 0.5) and (knee_vis > 0.5)

                if not is_visible:
                    # --- STATE: NOT READY ---
                    # dim the screen to show it's inactive
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                    
                    # Show Warning
                    cv2.putText(frame, "STEP BACK", (w//4, h//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    cv2.putText(frame, "Ensure full body is in frame", (w//4 - 50, h//2 + 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            
                else:
                    # --- STATE: ACTIVE (Run the Logic) ---
                    # MediaPipe IDs: 11=Left Shoulder, 13=Left Elbow, 15=Left Wrist
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
                    
                    # brain of the tracker
                    current_state = calculate_next_state(current_state, elbow_angle)
                    
                    #form check---
                    if hip_angle > 170:
                        status_color = (0, 255, 0) #green
                        feedback_text = "FORM: GOOD"
                    else:
                        status_color = (0, 0, 255) #red
                        feedback_text = "FIX HIPS!"

                    #updating total form errors
                    if feedback_text == "FIX HIPS!":
                        total_form_errors += 1
                    if hip_angle < worst_recorded_angle:
                        worst_recorded_angle = hip_angle

                    # --- AUDIO TRIGGER 1: REP COMPLETION ---
                    # If count jumped from 0 -> 1, speak "One"
                    if current_state.count > previous_count:
                        voice.speak(str(current_state.count), cooldown=0.5)
                        previous_count = current_state.count

                    # --- AUDIO TRIGGER 2: FORM CORRECTION ---
                    # Only speak errors if we are in the "active" zone (checking visibility)
                    if 'feedback_text' in locals() and feedback_text == "FIX HIPS!":
                        # 3 second cooldown so it doesn't spam you
                        voice.speak("Fix your hips", cooldown=3.0)    

                    #-----visualization-----
                    # syntax: cv2.putText(image, text, org, font, fontScale, color, thickness=1, lineType=cv2.LINE_8, bottomLeftOrigin=False)
                    # cv2.line(image, start_point, end_point, color, thickness)

                    #draw arm (yellow for tracking)
                    cv2.line(frame, (p_shoulder[0], p_shoulder[1]), (p_elbow[0], p_elbow[1]), (0, 255, 255), 3)
                    cv2.line(frame, (p_elbow[0], p_elbow[1]), (p_wrist[0], p_wrist[1]), (0, 255, 255), 3)

                    # draw body (color coded by form)
                    # shoulder -> hip -> knee
                    cv2.line(frame, (p_shoulder[0], p_shoulder[1]), (p_hip[0], p_hip[1]), status_color, 3)
                    cv2.line(frame, (p_hip[0], p_hip[1]), (p_knee[0], p_knee[1]), status_color, 3)

                    # draw circles at joints
                    cv2.circle(frame, (p_hip[0], p_hip[1]), 8, status_color, cv2.FILLED)


                    # UI
                    # rep counter box
                    cv2.rectangle(frame, (0,0), (300, 100), (245, 117, 16), cv2.FILLED)
                    cv2.putText(frame, f"REPS: {current_state.count}", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, f"{current_state.stage}", (10, 90), 
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
                    
                    # feedback text (only show if there is an error)
                    if feedback_text == "FIX HIPS!":
                        cv2.putText(frame, feedback_text, (50, 200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                        
                cv2.imshow("Frame", frame)  

                # CHECK FOR QUIT
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    workout_active = False # <--- Switch modes
                    voice.speak("Workout complete. Analyzing.")
    
        else:
            # === PART 2: THE SUMMARY SCREEN ===
            
            # darken the last frame to create a background
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

            # generate feedback
            if not summary_generated:
                # show 'loading' text
                cv2.putText(frame, "Consulting Coach Gemini...", (100, 300), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                cv2.imshow("Frame", frame)
                cv2.waitKey(1) # force update screen
                
                #call api
                if current_state.count > 0:
                    coach_feedback = ai_coach.generate_feedback(
                        reps=current_state.count,
                        errors=total_form_errors,
                        worst_angle=worst_recorded_angle
                    )
                else:
                    coach_feedback = "No reps recorded."

                summary_generated = True
                # speak it
                voice.speak(coach_feedback, cooldown=0)

                # === display the report card===
                cv2.putText(frame, "WORKOUT SUMMARY", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                cv2.putText(frame, f"Total Reps: {current_state.count}", (50, 160), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                cv2.putText(frame, f"Form Errors: {total_form_errors}", (50, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                
                # Draw the Gemini Text wrapped
                draw_text_with_background(frame, coach_feedback, (50, 300), scale=0.8)
                
                cv2.putText(frame, "Press 'Q' again to Exit", (50, 650), cv2.FONT_HERSHEY_PLAIN, 1.5, (100, 100, 100), 2)

                # EXIT CONDITION
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.imshow("Frame", frame)

    cap.release()
    cv2.destroyAllWindows()


# if run directly, main is executed. python internally sets __name__ = __main__ but if I'm importing it __name__ == __main__ is false
if __name__ == "__main__":
    main()


    