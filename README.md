# Calisthenics Form Tracker

A real-time Python application that uses a webcam to count exercise repetitions and give instant voice alerts if your form breaks down. 

---

## Addressed Challenges

* **Hard to Count Manually:** Simple trackers miss the distinct phases of a rep (going down, holding, coming back up). This app uses a state machine to make sure a rep is fully completed before counting it.
* **Bad Form Causes Injury:** The app checks for specific mistakes (like sagging hips during push-ups) instead of just counting volume.
* **Can't Look at a Screen Mid-Set:** Checking a screen ruins your posture. This app uses real-time audio alerts so you can keep your head away from screen and still fix your form.

---

## Built With

* **OpenCV:** Handles opening your webcam, reading video frames, and rendering the visual overlay.
* **MediaPipe:** Extracts 3D coordinate points of human joints from the video frames in real time.
* **Python Threading:** Runs the text-to-speech audio on a separate background thread so the video feed doesn't lag or stutter when the coach speaks.

---

## Project Structure

```text
├── pose_tracker.py       # Reads webcam feed via OpenCV and gets joint coordinates from MediaPipe
├── geometry_engine.py    # Uses math (dot products) to calculate exact joint angles from coordinates
├── rep_tracker.py        # Tracks rep states (Start -> Moving -> Peak -> Return) and flags form errors
├── voice_engine.py       # Speaks form corrections in the background without freezing the video
└── ai_coach.py           # The main script that glues the video, math, logic, and audio together
