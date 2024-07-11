import cv2
import face_recognition
import numpy as np
import h5py
import os
from multiprocessing import Pool

def load_trained_model_hdf5(model_path="trained_model.hdf5"):
    known_face_encodings = []
    known_face_names = []
    with h5py.File(model_path, 'r') as f:
        for person_name in f.keys():
            encodings = f[f"{person_name}/encodings"][:]
            names = f[f"{person_name}/names"][:]
            known_face_encodings.extend(encodings)
            known_face_names.extend([name.decode('utf-8') for name in names])
    return known_face_encodings, known_face_names
def process_frame(args):
    frame, known_face_encodings, known_face_names = args
    frame_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    face_locations = face_recognition.face_locations(frame_small)
    face_encodings = face_recognition.face_encodings(frame_small, face_locations)
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "New User"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        face_names.append(name)
    return frame, face_locations, face_names

def main():
    known_face_encodings, known_face_names = load_trained_model_hdf5()
    video_capture = cv2.VideoCapture(0)
    pool = Pool(processes=os.cpu_count())
    while True:
        ret, frame = video_capture.read()
        if not ret:break
        frame, face_locations, face_names = pool.map(process_frame, [(frame, known_face_encodings, known_face_names)])[0]
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top, right, bottom, left = top*4, right*4, bottom*4, left*4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
