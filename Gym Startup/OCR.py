import cv2
import pytesseract
from pytesseract import Output
import threading
import queue

frameQueue = queue.Queue()
textQueue = queue.Queue()

def ocr_thread():
    while True:
        if not frameQueue.empty():
            frame = frameQueue.get()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
            details = pytesseract.image_to_data(gray_frame, output_type=Output.DICT, config=custom_config, lang='eng')
            for i in range(len(details['text'])):
                if int(details['conf'][i]) > 30:
                    (x, y, w, h) = (details['left'][i], details['top'][i], details['width'][i], details['height'][i])
                    text = details['text'][i]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            textQueue.put(frame)

def webcam_thread():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frameQueue.empty():
            frameQueue.put(frame)
        if not textQueue.empty():
            cv2.imshow('Frame', textQueue.get())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

if __name__ == "__main__":
    threading.Thread(target=ocr_thread, daemon=True).start()
    webcam_thread()
    cv2.destroyAllWindows()
