import cv2
import mediapipe as mp
import numpy as np
from pymongo import MongoClient
from multiprocessing import Pool
import os
import time
import imutils

from face_run import load_trained_model_hdf5, process_frame

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

client = MongoClient('mongodb+srv://ayush21031:ayushsachan02@cluster0.vdxug01.mongodb.net/')
db = client['exercise_database']
exercise_data_collection = db['exercise_data']

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
}

arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT["DICT_5X5_100"])
arucoParams = cv2.aruco.DetectorParameters_create()

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

def is_correct_form(angle, correct_angle=180, correct_angle_range=None):
    if correct_angle_range:
        return correct_angle_range[0] <= angle <= correct_angle_range[1]
    else:
        lower_bound = correct_angle * 0.94
        upper_bound = correct_angle * 1.06
        return lower_bound <= angle <= upper_bound

def detect_back_bend(landmarks):
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

    angle_shoulder_hip_left = calculate_angle(left_shoulder, left_hip, left_knee)
    angle_shoulder_hip_right = calculate_angle(right_shoulder, right_hip, right_knee)

    return angle_shoulder_hip_left, angle_shoulder_hip_right

exercise_config = {
    'bicep_curl': {
        'angles': {
            'arm': [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST]
        },
        'down_range': (150, 170),
        'up_range': (35, 55),
        'back_bend_detection': False
    },
    'seated_shoulder_press': {
        'angles': {
            'left_arm': [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST],
            'right_arm': [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST]
        },
        'down_range': (80, 100),
        'up_range': (160, 180),
        'back_bend_detection': True
    },
    'lateral_raises': {
        'angles': {
            'left_arm': [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST],
            'right_arm': [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST]
        },
        'down_range': (70, 90),
        'up_range': (130, 150),
        'back_bend_detection': False
    }
}

class ExerciseAnalyzer:
    def __init__(self, exercise):
        self.exercise = exercise
        self.username = None
        self.reset_counters()

    def reset_counters(self):
        self.rep_count = 0
        self.set_count = 0
        self.is_down_complete = False
        self.is_up_complete = False
        self.stable_start_detected = False
        self.stable_frames = 0
        self.sets_reps = []
        self.rep_data = []
        self.rep_start_angle = None
        self.rep_end_angle = None
        self.current_weight = 0
        self.last_update_time = time.time()
        self.last_activity_time = None
        self.person_in_frame = False

    def analyze_exercise_form(self, landmarks, frame):
        feedback_texts = []
        activity_detected = False

        if exercise_config[self.exercise].get('back_bend_detection'):
            angle_shoulder_hip_left, angle_shoulder_hip_right = detect_back_bend(landmarks)

            if angle_shoulder_hip_left < 160 or angle_shoulder_hip_right < 160:
                # Back bend detected
                cv2.line(frame, (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1]), int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0])),
                         (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * frame.shape[1]), int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * frame.shape[0])), (0, 0, 255), 3)
                cv2.line(frame, (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1]), int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0])),
                         (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * frame.shape[1]), int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * frame.shape[0])), (0, 0, 255), 3)

        for angle_name, points in exercise_config[self.exercise]['angles'].items():
            p1 = [landmarks[points[0].value].x, landmarks[points[0].value].y]
            p2 = [landmarks[points[1].value].x, landmarks[points[1].value].y]
            p3 = [landmarks[points[2].value].x, landmarks[points[2].value].y]
            angle = calculate_angle(p1, p2, p3)

            corners, ids, rejected = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
            if ids is not None:
                self.current_weight = int(ids[0][0])

            if not self.stable_start_detected:
                if is_correct_form(angle, correct_angle_range=exercise_config[self.exercise]['down_range']):
                    self.stable_frames += 1
                    if self.stable_frames > 30:
                        self.stable_start_detected = True
                        self.last_activity_time = time.time()
                        feedback_texts.append("Ready to start!")
                else:
                    self.stable_frames = max(0, self.stable_frames - 1)
                continue

            if self.stable_start_detected:
                if is_correct_form(angle, correct_angle_range=exercise_config[self.exercise]['down_range']):
                    if not self.is_down_complete:
                        self.is_down_complete = True
                        self.rep_start_angle = angle
                        activity_detected = True
                elif is_correct_form(angle, correct_angle_range=exercise_config[self.exercise]['up_range']):
                    if self.is_down_complete:
                        self.is_up_complete = True
                        self.rep_end_angle = angle
                        activity_detected = True

                if self.is_down_complete and self.is_up_complete:
                    self.rep_count += 1
                    self.rep_data.append((self.rep_start_angle, self.rep_end_angle, self.current_weight))
                    self.is_down_complete = False
                    self.is_up_complete = False
                    self.last_update_time = time.time()

                if self.rep_count >= 12:
                    self.sets_reps.append(self.rep_count)
                    self.set_count += 1
                    self.rep_count = 0
                    self.last_update_time = time.time()

            feedback_text = f"{angle_name.replace('_', ' ').title()}: (Angle: {int(angle)}°)"
            feedback_texts.append(feedback_text)

        if self.current_weight is not None:
            feedback_texts.append(f"Detected Weight: {self.current_weight}")
        
        if activity_detected:
            self.last_activity_time = time.time()

        return feedback_texts

    def update_data(self):
        if self.rep_count > 0:
            self.sets_reps.append(self.rep_count)
        self.set_count = int(self.set_count)
        self.sets_reps = [int(rep) for rep in self.sets_reps]
        self.rep_data = [(int(start), int(end), int(weight)) for start, end, weight in self.rep_data]

        exercise_data_collection.update_one(
            {'username': self.username, 'exercise': self.exercise},
            {'$set': {
                'set_count': self.set_count,
                'sets_reps': self.sets_reps,
                'rep_data': self.rep_data,
            }},
            upsert=True
        )

def main():
    exercises = list(exercise_config.keys())
    print("Choose an exercise:")
    for i, exercise in enumerate(exercises, start=1):
        print(f"{i}. {exercise}")
    choice = int(input("Enter your choice (1-3): "))

    if choice not in range(1, 4):
        print("Invalid choice. Exiting...")
        return

    exercise = exercises[choice - 1]
    analyzer = ExerciseAnalyzer(exercise)
    known_face_encodings, known_face_names = load_trained_model_hdf5()
    pool = Pool(processes=os.cpu_count())
    cap_face = cv2.VideoCapture(4)
    cap_exercise = cv2.VideoCapture(2)
    cap_face.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap_face.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap_exercise.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap_exercise.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
  
    face_recognition_active = True
    last_face_recognized_time = None
    recognized_face_duration_threshold = 1.0

    try:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0) as pose:
            while True:
                ret_face, frame_face = cap_face.read()
                ret_exercise, frame_exercise = cap_exercise.read()
                if not ret_face or not ret_exercise:
                    print("Failed to capture frames from one or both cameras.")
                    break

                if face_recognition_active:
                    face_recognition_result = pool.apply_async(process_frame, [(frame_face, known_face_encodings, known_face_names)])
                    frame_face, face_locations, face_names = face_recognition_result.get()
                    if face_locations:
                        if last_face_recognized_time is None:
                            last_face_recognized_time = time.time()
                        elif time.time() - last_face_recognized_time >= recognized_face_duration_threshold:
                            face_recognition_active = False
                            analyzer.username = face_names[0]
                            print(f"Face recognized as {analyzer.username}, starting exercise analysis.")
                            cv2.destroyWindow('Face Recognition')
                    else:
                        last_face_recognized_time = None

                    for (top, right, bottom, left), name in zip(face_locations, face_names):
                        top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
                        cv2.rectangle(frame_face, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.putText(frame_face, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.imshow('Face Recognition', frame_face)

                if not face_recognition_active:
                    cv2.destroyWindow('Face Recognition')
                    frame_exercise_rgb = cv2.cvtColor(frame_exercise, cv2.COLOR_BGR2RGB)
                    results = pose.process(frame_exercise_rgb)
                    frame_exercise = cv2.cvtColor(frame_exercise_rgb, cv2.COLOR_RGB2BGR)
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(frame_exercise, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                  mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3, circle_radius=2),
                                                  mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
                        
                        landmarks = results.pose_landmarks.landmark
                        
                        frame_exercise_landmarks = np.zeros_like(frame_exercise)
                        
                        mp_drawing.draw_landmarks(frame_exercise_landmarks, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                  mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3, circle_radius=2),
                                                  mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))
                        
                        overlay = cv2.addWeighted(frame_exercise, 0.5, frame_exercise_landmarks, 0.5, 0)

                        feedback_texts = analyzer.analyze_exercise_form(landmarks, frame_exercise)
                        for i, text in enumerate(feedback_texts):
                            cv2.putText(frame_exercise_landmarks, text, (10, 30 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        rep_set_text = f"Reps: {analyzer.rep_count} | Sets: {analyzer.set_count}"
                        cv2.putText(frame_exercise_landmarks, rep_set_text, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        cv2.imshow('Exercise Landmarks', frame_exercise_landmarks)
                        cv2.imshow('Exercise Detection', overlay)
                    
                    else:
                        if analyzer.person_in_frame:  
                            print("Person exited the frame, updating database...")
                            face_recognition_active = True
                            analyzer.update_data()
                            analyzer.reset_counters()
                            cv2.destroyWindow('Exercise Landmarks')
                            cv2.destroyWindow('Exercise Detection')
                        analyzer.person_in_frame = False

                    if analyzer.last_activity_time and time.time() - analyzer.last_activity_time >= 15:
                        print("No activity detected for 15 seconds, resetting...")
                        face_recognition_active = True
                        analyzer.update_data()
                        analyzer.reset_counters()
                        cv2.destroyWindow('Exercise Landmarks')
                        cv2.destroyWindow('Exercise Detection')
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        cap_face.release()
        cap_exercise.release()
        cv2.destroyAllWindows()
        pool.close()
        pool.join()
        analyzer.update_data()

if __name__ == "__main__":
    main()
