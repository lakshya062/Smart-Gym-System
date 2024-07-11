import cv2

def check_cameras(num_ports):
    for port in range(num_ports):
        camera = cv2.VideoCapture(port)
        if not camera.isOpened():
            print(f"No camera detected on port {port}!")
        else:
            print(f"Camera detected on port {port}!")
            camera.release()

num_ports = 10
check_cameras(num_ports)
