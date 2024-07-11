import cv2

# Load the face detection model (you need to provide this)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# The actual width of a face (in inches)
KNOWN_WIDTH = 6.0

# The focal length of the camera (this needs to be calibrated)
FOCAL_LENGTH = 615

def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / perWidth

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw rectangle around the faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Calculate distance
        distance = distance_to_camera(KNOWN_WIDTH, FOCAL_LENGTH, w)

        # Display distance on frame
        cv2.putText(frame, f"{distance:.2f} inches", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('img', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
