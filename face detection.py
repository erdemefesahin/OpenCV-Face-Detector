import cv2
import sys


def detect_faces(image_path: str, output_path: str | None = None) -> int:
    """Detect all faces in an image, draw rectangles, show and optionally save the result.

    Returns the number of faces detected.
    """
    # use multiple cascades to improve recall (frontal, alt2, profile)
    face_cascade1 = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    face_cascade2 = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
    )
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    # use eye cascade as a verification step so we primarily keep human faces
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    # fallback cascade that works better with eyeglasses/sunglasses
    eye_glasses_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
    )

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: couldn't read image at '{image_path}'")
        return 0

    # convert to gray and improve contrast to help detection in varied lighting
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # tune parameters for better multiple-face detection
    faces1 = face_cascade1.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(30, 30))
    faces2 = face_cascade2.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(30, 30))
    profiles = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    # profile cascade detects faces looking to one side only; run on flipped image too
    gray_flipped = cv2.flip(gray, 1)
    profiles_flipped = profile_cascade.detectMultiScale(gray_flipped, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    # map flipped coordinates back to original image
    h_img, w_img = gray.shape[:2]
    profiles_mapped = []
    for (x, y, w, h) in profiles_flipped:
        xf = w_img - x - w
        profiles_mapped.append((xf, y, w, h))
    # combine original and flipped profile detections
    profiles = list(profiles) + profiles_mapped

    # mark source type so we can relax verification for profile detections
    all_candidates = []
    for (x, y, w, h) in list(faces1) + list(faces2):
        all_candidates.append((x, y, w, h, 'frontal'))
    for (x, y, w, h) in profiles:
        all_candidates.append((x, y, w, h, 'profile'))

    # simple NMS: merge overlapping rectangles to reduce duplicates
    def nms(boxes, iou_threshold=0.3):
        if len(boxes) == 0:
            return []
        rects = [ [x, y, x+w, y+h, 1.0, src] for (x,y,w,h,src) in boxes ]
        # convert to arrays
        boxes_arr = []
        for r in rects:
            boxes_arr.append(r)
        boxes_arr = sorted(boxes_arr, key=lambda b: b[4], reverse=True)
        picked = []
        while boxes_arr:
            current = boxes_arr.pop(0)
            picked.append(current)
            rest = []
            for b in boxes_arr:
                # compute IoU
                x1 = max(current[0], b[0])
                y1 = max(current[1], b[1])
                x2 = min(current[2], b[2])
                y2 = min(current[3], b[3])
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                inter = w * h
                area1 = (current[2]-current[0]) * (current[3]-current[1])
                area2 = (b[2]-b[0]) * (b[3]-b[1])
                iou = inter / float(area1 + area2 - inter + 1e-6)
                if iou <= iou_threshold:
                    rest.append(b)
            boxes_arr = rest
        # convert back to (x,y,w,h,src)
        out = []
        for p in picked:
            src = p[5]
            out.append((int(p[0]), int(p[1]), int(p[2]-p[0]), int(p[3]-p[1]), src))
        return out

    merged = nms(all_candidates, iou_threshold=0.35)

    accepted = []
    for (x, y, w, h, src) in merged:
        # simple aspect ratio/size checks to avoid long thin objects (ties, hands)
        aspect = w / float(h) if h > 0 else 0
        if w < 30 or h < 30:
            continue
        if aspect < 0.4 or aspect > 1.6:
            continue

        # verify by checking for eyes inside the face ROI
        roi_gray = gray[y : y + h, x : x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
        if len(eyes) == 0:
            # try glasses-tolerant eye detector
            eyes = eye_glasses_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
        # if source is profile, accept if eyes OR profile detection (relax)
        if src == 'profile':
            if len(eyes) == 0:
                # still accept profile detection (eyes may not be visible)
                accepted.append((x, y, w, h))
                continue
        else:
            # frontal detection: require at least one eye
            if len(eyes) == 0:
                continue
            accepted.append((x, y, w, h))

    # pick a palette of distinct colors
    palette = [
        (0, 255, 0),
        (0, 128, 255),
        (255, 0, 0),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 128),
        (0, 128, 0),
    ]

    for i, (x, y, w, h) in enumerate(accepted, start=1):
        color = palette[(i - 1) % len(palette)]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
        # draw filled label background for readability
        label = f"Person {i}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x, y - th - 12), (x + tw + 6, y), color, -1)
        cv2.putText(img, label, (x + 3, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Show the image with all detected faces at once
    cv2.imshow('face detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # default output path if none provided
    if output_path is None:
        output_path = 'annotated_' + image_path

    cv2.imwrite(output_path, img)

    return len(accepted)


if __name__ == '__main__':
    # allow user to pass image path and optional output path
    img_path = 'image5.jpeg'
    out_path = None
    if len(sys.argv) >= 2:
        img_path = sys.argv[1]
    if len(sys.argv) >= 3:
        out_path = sys.argv[2]

    count = detect_faces(img_path, out_path)
    print(f"Detected {count} face(s) in '{img_path}'")