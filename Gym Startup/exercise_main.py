import cv2
import numpy as np
import mediapipe as mp
from exercise_analyzer import ExerciseAnalyzer

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

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

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    exercise = 'bicep_curl'
    analyzer = ExerciseAnalyzer(exercise, exercise_config)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                feedback_texts = analyzer.analyze_exercise_form(landmarks)
                for i, text in enumerate(feedback_texts):
                    cv2.putText(frame, text, (10, 30 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                rep_set_text = f"Reps: {analyzer.rep_count} | Sets: {analyzer.set_count}"
                cv2.putText(frame, rep_set_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.imshow('Exercise Form', frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
