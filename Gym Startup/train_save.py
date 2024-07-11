import cv2
import face_recognition
import os
import numpy as np
import h5py

def extract_face_encodings(video_path, max_frames_to_extract=5):
    video_capture = cv2.VideoCapture(video_path)
    known_face_encodings = []
    frame_count = 0
    while frame_count < max_frames_to_extract:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        face_locations = face_recognition.face_locations(frame_small)
        face_encodings = face_recognition.face_encodings(frame_small, face_locations)
        known_face_encodings.extend(face_encodings)
        frame_count += 1
    video_capture.release()
    return known_face_encodings

def train_and_save_model_hdf5(video_clips_directory, save_path="trained_model.hdf5"):
    with h5py.File(save_path, 'w') as f:
        for video_clip_filename in os.listdir(video_clips_directory):
            if video_clip_filename.endswith(".mp4"):
                person_name = os.path.splitext(video_clip_filename)[0]
                video_path = os.path.join(video_clips_directory, video_clip_filename)
                face_encodings = extract_face_encodings(video_path)
                if not face_encodings:  # Skip if no faces found
                    continue
                ds_encodings = f.create_dataset(f"{person_name}/encodings", data=np.array(face_encodings), compression="gzip")
                ds_names = f.create_dataset(f"{person_name}/names", data=np.array([person_name] * len(face_encodings), dtype='S'), compression="gzip")

if __name__ == "__main__":
    video_clips_directory = "Video_clips/"
    train_and_save_model_hdf5(video_clips_directory)
