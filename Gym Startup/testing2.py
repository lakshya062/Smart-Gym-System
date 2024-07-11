import cv2
import pickle

def load_trained_model_and_labels():
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('trained_face_recognizer.yml')

    with open('labels.pickle', 'rb') as f:
        names = pickle.load(f)

    return face_recognizer, names   
def recognize_face(face_recognizer, names, frame, gray, x, y, w, h):
    id, confidence = face_recognizer.predict(gray[y:y+h, x:x+w])
    if confidence < 100:
        person_name = names[id]
        confidence_text = "  {0}%".format(round(100 - confidence))
    else:
        person_name = "Unknown"
        confidence_text = "  {0}%".format(round(100 - confidence))

    return person_name, confidence_text, (x, y, w, h)
def live_recognition():
    face_recognizer, names = load_trained_model_and_labels()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            person_name, confidence_text, coords = recognize_face(face_recognizer, names, frame, gray, x, y, w, h)
            cv2.putText(frame, f"{person_name}{confidence_text}", (coords[0]+5, coords[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.rectangle(frame, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (255, 0, 0), 2)

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

live_recognition()
