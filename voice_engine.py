import pyttsx3
import threading
import time

class VoiceEngine:
    def __init__(self, rate=170):
        self.rate = rate
        self.last_spoken_time = 0
        self.lock = threading.Lock()

    def speak(self, text, cooldown=3.0):
        """
        Says the text, but only if cooldown seconds have passed since last speech.
        runs in a separate thread to prevent video freezing.
        """
        current_time = time.time()
        if current_time - self.last_spoken_time > cooldown:
            self.last_spoken_time = current_time
            
            t = threading.Thread(target=self._speak_worker, args=(text,))
            t.start()

    def _speak_worker(self, text):
        try:
            # we create a new engine instance for every thread
            engine = pyttsx3.init()
            engine.setProperty('rate', self.rate)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Voice Error: {e}")
