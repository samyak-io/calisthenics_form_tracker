import google.generativeai as genai
import os 
from dotenv import load_dotenv

load_dotenv()

class AICoach:
    def __init__(self):
        
        self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            print("WARNING: No API Key found in .env file. Coach is disabled")
            return
        
        try:
            genai.configure(api_key = self.api_key)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
        except Exception as e:
            print("Error configuring GOOGLE AI: {e}")
        
    def generate_feedback(self, reps, errors, worst_angle):
        if not self.api_key:
            return "Skipping AI feedback (No Key)."
        
        print("\n Connecting to Gemini... (Please wait)")

        prompt = f"""
        You are an elite Calisthenics Coach. I just finished a set of pushups.
        Stats:
        - Reps: {reps}
        - Hip Sag Errors: {errors}
        - Worst Angle: {int(worst_angle)} degrees (Ideal 180)

        Write a short spoken response for me.
        1. NO Markdown, NO asterisks, NO special characters.
        2. Keep it under 3 sentences.
        3. Be direct and specific.
        4. Write exactly what you would SAY.
        """

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Coach Connection Failed: {e}"