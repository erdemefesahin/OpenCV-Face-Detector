import cv2
import sys
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog

"""
Optional dependency: DeepFace for age/gender with auto-downloaded weights.
If not installed, the script falls back to Caffe age/gender models when available.
"""
try:
    from deepface import DeepFace  # type: ignore
    _HAS_DEEPFACE = True
except Exception:
    DeepFace = None
    _HAS_DEEPFACE = False

# One-time console notices for DeepFace downloads/errors
_DF_NOTICE_SHOWN = False
_DF_FAIL_SHOWN = False


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']


def load_face_detector(model_dir):
    proto_path = os.path.join(model_dir, 'deploy.prototxt')
    model_path = os.path.join(model_dir, 'res10_300x300_ssd_iter_140000.caffemodel')
    if os.path.exists(proto_path) and os.path.exists(model_path):
        try:
            net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
            return net
        except Exception as e:
            print('DNN face model load failed, falling back to Haar:', e)
    return None


def detect_faces(img, dnn_net=None, conf_thresh=0.45):
    faces = []
    (h, w) = img.shape[:2]
    if dnn_net is not None:
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        dnn_net.setInput(blob)
        detections = dnn_net.forward()
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
    if not faces:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = list(face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(60, 60)))
    return faces


def load_age_gender_models(model_dir):
    age_proto = os.path.join(model_dir, 'age_deploy.prototxt')
    age_model = os.path.join(model_dir, 'age_net.caffemodel')
    gender_proto = os.path.join(model_dir, 'gender_deploy.prototxt')
    gender_model = os.path.join(model_dir, 'gender_net.caffemodel')

    age_net = None
    gender_net = None

    try:
        if os.path.exists(age_proto) and os.path.exists(age_model):
            age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
        else:
            # Suppress noisy warning if DeepFace is available (it will provide age)
            if not _HAS_DEEPFACE:
                print("Age model files not found in 'models/'. Age labels will be skipped.")
    except Exception as e:
        print('Failed to load age model:', e)

    try:
        if os.path.exists(gender_proto) and os.path.exists(gender_model):
            gender_net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)
        else:
            if not _HAS_DEEPFACE:
                print("Gender model files not found in 'models/'. Gender labels will be skipped.")
    except Exception as e:
        print('Failed to load gender model:', e)

    return age_net, gender_net


# Removed URL-based downloader since public links are unreliable


def classify_age_gender(face_roi_bgr, age_net, gender_net):
    # Prepare blob for age/gender networks (Caffe, BGR mean values)
    blob = cv2.dnn.blobFromImage(face_roi_bgr, scalefactor=1.0, size=(227, 227),
                                 mean=MODEL_MEAN_VALUES, swapRB=False, crop=False)
    gender_label = None
    age_bucket = None
    gender_conf = None
    age_conf = None

    if gender_net is not None:
        gender_net.setInput(blob)
        gender_preds = gender_net.forward().flatten()
        g_idx = int(np.argmax(gender_preds))
        gender_label = GENDER_LIST[g_idx]
        gender_conf = float(gender_preds[g_idx])

    if age_net is not None:
        age_net.setInput(blob)
        age_preds = age_net.forward().flatten()
        a_idx = int(np.argmax(age_preds))
        age_bucket = AGE_BUCKETS[a_idx]
        age_conf = float(age_preds[a_idx])

    return gender_label, gender_conf, age_bucket, age_conf


def classify_age_gender_deepface(face_roi_bgr):
    """Classify age and gender using DeepFace if available.
    Returns (gender_label, age_int) or (None, None) if unavailable.
    """
    if not _HAS_DEEPFACE or face_roi_bgr is None or face_roi_bgr.size == 0:
        return None, None
    try:
        global _DF_NOTICE_SHOWN, _DF_FAIL_SHOWN
        if not _DF_NOTICE_SHOWN:
            print("DeepFace: initializing age/gender (first run may download models; please wait)...")
            _DF_NOTICE_SHOWN = True
        roi_rgb = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2RGB)
        res = DeepFace.analyze(roi_rgb, actions=['age', 'gender'], enforce_detection=False, prog_bar=False)
        if isinstance(res, list) and len(res) > 0:
            res = res[0]
        age_val = None
        gender_label = None
        if isinstance(res, dict):
            # Age
            if 'age' in res and res['age'] is not None:
                try:
                    age_val = int(round(float(res['age'])))
                except Exception:
                    age_val = None
            # Gender label
            dom = res.get('dominant_gender') or res.get('gender')
            if isinstance(dom, str):
                low = dom.lower()
                if low in ('man', 'male'):
                    gender_label = 'Male'
                elif low in ('woman', 'female'):
                    gender_label = 'Female'
                else:
                    gender_label = dom.title()
        return gender_label, age_val
    except Exception as e:
        if not _DF_FAIL_SHOWN:
            print(f"DeepFace: analyze failed, falling back if available. Details: {e}")
            _DF_FAIL_SHOWN = True
        return None, None


def age_bucket_to_group(age_bucket):
    if not age_bucket:
        return None
    # Parse the upper bound from strings like '(25-32)'
    digits = ''.join(ch if ch.isdigit() else ' ' for ch in age_bucket).split()
    try:
        upper = int(digits[-1]) if digits else 0
    except Exception:
        upper = 0
    return 'Young' if upper <= 32 else 'Old'


def age_bucket_to_range(age_bucket):
    # Convert '(25-32)' -> (25, 32); returns (None, None) on failure
    if not age_bucket:
        return (None, None)
    digits = ''.join(ch if ch.isdigit() else ' ' for ch in age_bucket).split()
    if len(digits) >= 2:
        try:
            return int(digits[0]), int(digits[1])
        except Exception:
            return (None, None)
    return (None, None)


def age_bucket_midpoint(age_bucket):
    lo, hi = age_bucket_to_range(age_bucket)
    if lo is None or hi is None:
        return None
    return int(round((lo + hi) / 2))


def draw_label(img, x, y, w, h, color, text):
    # Put text above the bounding box with a filled background for readability
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    box_x1 = x
    box_y1 = max(0, y - th - baseline - 6)
    box_x2 = x + tw + 8
    box_y2 = y
    cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), color, -1)
    cv2.putText(img, text, (x + 4, box_y2 - 6), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def annotate_and_draw(img, faces, age_net, gender_net, palette):
    male_count = 0
    female_count = 0
    for i, (x, y, w, h) in enumerate(faces):
        color = palette[i % len(palette)]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        label_text = 'Face'
        try:
            # Crop ROI once
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
            roi = img[y1:y2, x1:x2]

            parts = []
            # 1) Try DeepFace (auto-downloads models)
            g_df, age_df = classify_age_gender_deepface(roi)
            if g_df:
                parts.append(g_df)
                if g_df == 'Male':
                    male_count += 1
                elif g_df == 'Female':
                    female_count += 1
            if age_df is not None:
                parts.append(f'Age {age_df}')

            # 2) Fallback to Caffe age/gender if nothing added
            if not parts and (age_net is not None or gender_net is not None) and roi.size != 0:
                roi_resized = cv2.resize(roi, (227, 227))
                g_label, g_conf, a_bucket, a_conf = classify_age_gender(roi_resized, age_net, gender_net)
                if g_label:
                    parts.append(g_label)
                    if g_label == 'Male':
                        male_count += 1
                    elif g_label == 'Female':
                        female_count += 1
                if a_bucket:
                    lo, hi = age_bucket_to_range(a_bucket)
                    if lo is not None and hi is not None:
                        parts.append(f'Age {lo}-{hi}')
                    else:
                        age_group = age_bucket_to_group(a_bucket)
                        if age_group:
                            parts.append(age_group)

            if parts:
                label_text = ', '.join(parts)
        except Exception:
            # Classification failure shouldn't break drawing
            pass

        draw_label(img, x, y, w, h, color, label_text)

    return male_count, female_count


def process_image(image_path, dnn_face_net, age_net, gender_net):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: couldn't read image at '{image_path}'")
        return

    faces = detect_faces(img, dnn_net=dnn_face_net, conf_thresh=0.45)

    palette = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
    ]
    male_count, female_count = annotate_and_draw(img, faces, age_net, gender_net, palette)

    cv2.imshow('face detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Detected {len(faces)} face(s) in '{image_path}'")
    if male_count or female_count:
        print(f"Breakdown: {male_count} male, {female_count} female")


def process_video(dnn_face_net, age_net, gender_net, cam_index=0):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print('Error: Could not open camera.')
        return

    palette = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
    ]

    print("Press 'q' or ESC to exit the video.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame, dnn_net=dnn_face_net, conf_thresh=0.5)
        annotate_and_draw(frame, faces, age_net, gender_net, palette)

        cv2.imshow('Live Face Detection', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    # Load models once
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    dnn_face_net = load_face_detector(model_dir)
    age_net, gender_net = load_age_gender_models(model_dir)

    if _HAS_DEEPFACE:
        print("Age/Gender: Using DeepFace (models auto-download on first use)")
    else:
        # If DeepFace not installed and no Caffe models loaded, warn user.
        if age_net is None and gender_net is None:
            print("Tip: Install DeepFace for automatic age/gender models: pip install deepface")

    # If a path was provided via CLI, process image directly
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        process_image(image_path, dnn_face_net, age_net, gender_net)
        return

    # GUI with two buttons: Photo and Video
    root = tk.Tk()
    root.title('Choose Mode')
    root.geometry('280x120')

    def on_photo():
        filetypes = [
            ('Image files', '*.png;*.jpg;*.jpeg;*.bmp;*.gif'),
            ('All files', '*.*'),
        ]
        path = filedialog.askopenfilename(title='Select an image', filetypes=filetypes)
        root.destroy()
        if path:
            process_image(path, dnn_face_net, age_net, gender_net)
        else:
            print('No image selected.')

    def on_video():
        root.destroy()
        process_video(dnn_face_net, age_net, gender_net, cam_index=0)

    btn_photo = tk.Button(root, text='Photo', width=18, command=on_photo)
    btn_video = tk.Button(root, text='Video', width=18, command=on_video)
    btn_photo.pack(pady=8)
    btn_video.pack(pady=6)

    root.mainloop()


if __name__ == '__main__':
    main()