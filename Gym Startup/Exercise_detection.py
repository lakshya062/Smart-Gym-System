import cv2
import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
def calculate_angle(a, b, c):
    a = np.array(a) 
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:angle = 360 - angle
    return angle
exercise_config = {
    'plank': {
        'angles': {
            'back_hips_legs': [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE],
            'shoulder_arm': [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST]
        },
        'correct_angle': 180
    },
    'bicep_curl': {
        'angles': {
            'arm': [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST]
        },
        'down_range': (150, 170),
        'up_range': (35, 55),
    }
}
def is_correct_form(angle, correct_angle=180, correct_angle_range=None):
    if correct_angle_range:return correct_angle_range[0] <= angle <= correct_angle_range[1]
    else:
        lower_bound = correct_angle * 0.94
        upper_bound = correct_angle * 1.06
        return lower_bound <= angle <= upper_bound
class ExerciseAnalyzer:
    def __init__(self, exercise):
        self.exercise = exercise
        self.rep_count = 0
        self.set_count = 0
        self.is_down_complete = False
        self.is_up_complete = False
        self.stable_start_detected = False
        self.stable_frames = 0
    def analyze_exercise_form(self, landmarks):
        feedback_texts = []
        for angle_name, points in exercise_config[self.exercise]['angles'].items():
            p1 = [landmarks[points[0].value].x, landmarks[points[0].value].y]
            p2 = [landmarks[points[1].value].x, landmarks[points[1].value].y]
            p3 = [landmarks[points[2].value].x, landmarks[points[2].value].y]
            angle = calculate_angle(p1, p2, p3)
            if not self.stable_start_detected:
                if is_correct_form(angle, correct_angle_range=exercise_config[self.exercise]['down_range']):
                    self.stable_frames += 1
                    if self.stable_frames > 30:
                        self.stable_start_detected = True
                        feedback_texts.append("Ready to start!")
                else:self.stable_frames = max(0, self.stable_frames - 1)
                continue
            if self.exercise == 'bicep_curl' and self.stable_start_detected:
                if is_correct_form(angle, correct_angle_range=exercise_config[self.exercise]['down_range']):self.is_down_complete = True
                elif is_correct_form(angle, correct_angle_range=exercise_config[self.exercise]['up_range']) and self.is_down_complete:self.is_up_complete = True
                if self.is_down_complete and self.is_up_complete:
                    self.rep_count += 1
                    self.is_down_complete = False
                    self.is_up_complete = False
                if self.rep_count >= 12:
                    self.set_count += 1
                    self.rep_count = 0
            correct_form = is_correct_form(angle, correct_angle=exercise_config[self.exercise].get('correct_angle', 180),correct_angle_range=exercise_config[self.exercise].get('correct_angle_range'))
            feedback_text = f"{angle_name.replace('_', ' ').title()}: {'Correct' if correct_form else 'Incorrect'} (Angle: {int(angle)}°)"
            feedback_texts.append(feedback_text)
        return feedback_texts
def main():
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    exercise = 'bicep_curl'
    analyzer = ExerciseAnalyzer(exercise)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:break
            frame = cv2.resize(frame, (640, 480))
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                feedback_texts = analyzer.analyze_exercise_form(landmarks)
                for i, text in enumerate(feedback_texts):cv2.putText(frame, text, (10, 30 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                rep_set_text = f"Reps: {analyzer.rep_count} | Sets: {analyzer.set_count}"
                cv2.putText(frame, rep_set_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.imshow('Exercise Form', frame)
                if cv2.waitKey(1) & 0xFF == 27:break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()