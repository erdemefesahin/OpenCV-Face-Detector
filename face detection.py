import cv2
import sys
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog


def main():
   
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        try:
            root = tk.Tk()
            root.withdraw()
            filetypes = [
                ('Image files', '*.png;*.jpg;*.jpeg;*.bmp;*.gif'),
                ('All files', '*.*'),
            ]
            image_path = filedialog.askopenfilename(title='Select an image', filetypes=filetypes)
            root.destroy()
        except Exception as e:
            print(f'File dialog error: {e}')
            return

    if not image_path:
        print('No image selected. Exiting.')
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: couldn't read image at '{image_path}'")
        return

    
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    proto_path = os.path.join(model_dir, 'deploy.prototxt')
    model_path = os.path.join(model_dir, 'res10_300x300_ssd_iter_140000.caffemodel')

    faces = []
    if os.path.exists(proto_path) and os.path.exists(model_path):
        try:
            net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
            (h, w) = img.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            conf_thresh = 0.45
            for i in range(0, detections.shape[2]):
                confidence = float(detections[0, 0, i, 2])
                if confidence < conf_thresh:
                    continue
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype('int')
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                ww, hh = x2 - x1, y2 - y1
                if ww > 0 and hh > 0:
                    faces.append((x1, y1, ww, hh))
        except Exception as e:
            print('DNN load failed, falling back to Haar:', e)

    # Fallback to Haar if no DNN faces were found or model missing
    if not faces:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = list(face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(60, 60)))

    # Color palette for different faces
    palette = [
        (0, 255, 0),      # Green
        (255, 0, 0),      # Blue
        (0, 0, 255),      # Red
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Yellow
        (128, 0, 128),    # Purple
        (255, 165, 0),    # Orange
    ]

    for i, (x, y, w, h) in enumerate(faces):
        color = palette[i % len(palette)]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    cv2.imshow('face detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Detected {len(faces)} face(s) in '{image_path}'")


if __name__ == '__main__':
    main()