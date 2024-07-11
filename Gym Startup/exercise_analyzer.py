import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a) 
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

class ExerciseAnalyzer:
    def __init__(self, exercise, exercise_config):
        self.exercise = exercise
        self.exercise_config = exercise_config
        self.rep_count = 0
        self.set_count = 0
        self.is_down_complete = False
        self.is_up_complete = False
        self.stable_start_detected = False
        self.stable_frames = 0

    def is_correct_form(self, angle, correct_angle=180, correct_angle_range=None):
        if correct_angle_range:
            return correct_angle_range[0] <= angle <= correct_angle_range[1]
        else:
            lower_bound = correct_angle * 0.94
            upper_bound = correct_angle * 1.06
            return lower_bound <= angle <= upper_bound

    def analyze_exercise_form(self, landmarks):
        feedback_texts = []
        for angle_name, points in self.exercise_config[self.exercise]['angles'].items():
            p1 = [landmarks[points[0].value].x, landmarks[points[0].value].y]
            p2 = [landmarks[points[1].value].x, landmarks[points[1].value].y]
            p3 = [landmarks[points[2].value].x, landmarks[points[2].value].y]
            angle = calculate_angle(p1, p2, p3)
            if not self.stable_start_detected:
                if self.is_correct_form(angle, correct_angle_range=self.exercise_config[self.exercise]['down_range']):
                    self.stable_frames += 1
                    if self.stable_frames > 30:
                        self.stable_start_detected = True
                        feedback_texts.append("Ready to start!")
                else:
                    self.stable_frames = max(0, self.stable_frames - 1)
                continue
            if self.exercise == 'bicep_curl' and self.stable_start_detected:
                if self.is_correct_form(angle, correct_angle_range=self.exercise_config[self.exercise]['down_range']):
                    self.is_down_complete = True
                elif self.is_correct_form(angle, correct_angle_range=self.exercise_config[self.exercise]['up_range']) and self.is_down_complete:
                    self.is_up_complete = True
                if self.is_down_complete and self.is_up_complete:
                    self.rep_count += 1
                    self.is_down_complete = False
                    self.is_up_complete = False
                if self.rep_count >= 12:
                    self.set_count += 1
                    self.rep_count = 0
            correct_form = self.is_correct_form(angle, correct_angle=self.exercise_config[self.exercise].get('correct_angle', 180), correct_angle_range=self.exercise_config[self.exercise].get('correct_angle_range'))
            feedback_text = f"{angle_name.replace('_', ' ').title()}: {'Correct' if correct_form else 'Incorrect'} (Angle: {int(angle)}°)"
            feedback_texts.append(feedback_text)
        return feedback_texts
