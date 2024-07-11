import cv2
import face_recognition
import numpy as np
from multiprocessing import Pool
import os
def load_trained_model(model_path="trained_model.npz"):
    data = np.load(model_path)
    known_face_encodings = data['face_encodings']
    known_face_names = data['names']
    return known_face_encodings, known_face_names
def process_frame(args):
    frame, known_face_encodings, known_face_names = args
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:name = known_face_names[best_match_index]
        face_names.append(name)
    return frame, face_locations, face_names
def main():
    known_face_encodings, known_face_names = load_trained_model()
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    pool = Pool(processes=os.cpu_count())
    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if frame_count % 10 == 0:frame, face_locations, face_names = pool.map(process_frame, [(frame, known_face_encodings, known_face_names)])[0]
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):break
        frame_count += 1
    video_capture.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":main()
