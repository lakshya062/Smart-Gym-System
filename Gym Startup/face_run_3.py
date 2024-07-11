import cv2
import mediapipe as mp
import face_recognition
import numpy as np
from multiprocessing import Pool
import os
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
def load_trained_model(model_path="trained_model.npz"):
    data = np.load(model_path)
    known_face_encodings = data['face_encodings']
    known_face_names = data['names']
    return known_face_encodings, known_face_names
def process_frame_for_face_recognition(frame, known_face_encodings, known_face_names):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:name = known_face_names[best_match_index]
        face_names.append(name)
    return face_locations, face_names
def main():
    known_face_encodings, known_face_names = load_trained_model()
    cap = cv2.VideoCapture(0)
    frame_count = 0
    face_locations = []
    face_names = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark
                finger_tips = [landmarks[i] for i in [8, 12, 16, 20]]
                for lm in hand_landmarks.landmark:
                    height, width, _ = frame.shape
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                num_extended_fingers = sum(1 for tip in finger_tips if tip.y < landmarks[5].y)
                cv2.putText(frame, f"Fingers: {num_extended_fingers}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if frame_count % 10 == 0:face_locations, face_names = process_frame_for_face_recognition(frame, known_face_encodings, known_face_names)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
        cv2.imshow('Hand Gesture and Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):break
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
