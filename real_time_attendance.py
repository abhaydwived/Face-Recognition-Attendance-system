import cv2  # Import OpenCV library for image and video processing
import face_recognition  # Import face_recognition for facial recognition tasks
import pickle  # Import pickle for loading saved data (face encodings)
import os  # Import os for file and directory operations
import csv  # Import csv for reading and writing CSV files
import numpy as np  # Import numpy for numerical operations
from datetime import datetime  # Import datetime to work with dates and times
from database.database_utils import log_attendance  # Import custom function to log attendance in the database
from spoof_utils import LivenessDetector  # Import liveness detection class to prevent spoofing

'''
Configuration section - These are the main settings for the attendance system.
Think of these as the "control panel" for how the system behaves.
'''
RE_LOG_GAP = 60 * 2  # Minimum seconds between logging the same person's attendance again
MIN_CONFIDENCE = 0.45  # Threshold for face recognition confidence (lower = stricter match required)
CSV_FILE = 'attendance_log.csv'  # Path to the CSV file where attendance is logged
YUNET_MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"  # Path to the YuNet face detection model file
ENABLE_LIVENESS_DETECTION = False  # Set True only when modelrgb.onnx is confirmed calibrated for your webcam
LIVENESS_THRESHOLD = 0.35  # Threshold for deciding if a face is real or spoofed
DEBUG_LIVENESS = False  # Set True only for debugging the liveness model

'''
Loading face encodings section:
This part loads the saved face data (encodings) that represent each person's face.
'''
try:
    with open('encodings/face_encodings.pkl', 'rb') as f:
        data = pickle.load(f)
    print(f"[INFO] Loaded {len(data['encodings'])} face encodings.")
    print(f"[INFO] Known names: {set(data['names'])}")
except Exception as e:
    print(f"[ERROR] Failed to load encodings: {e}")
    exit(1)

'''
YuNet face detector initialization:
YuNet is a fast and accurate face detection model that finds faces in images.
'''
yunet = None
try:
    dummy_cap = cv2.VideoCapture(0)
    ret, dummy_frame = dummy_cap.read()
    dummy_cap.release()
    if not ret or dummy_frame is None:
        raise Exception("No camera available to get frame shape for YuNet.")
    h, w = dummy_frame.shape[:2]
    yunet = cv2.FaceDetectorYN_create(
        YUNET_MODEL_PATH, "", (w, h), score_threshold=0.9, nms_threshold=0.3, top_k=5000
    )
    print("[INFO] YuNet face detector loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load YuNet model: {e}")

'''
Camera initialization function:
This function tries to find and connect to any available camera.
'''
def initialize_camera():
    camera_indices = [0, 1, 2, 3, 4]
    for idx in camera_indices:
        print(f"[INFO] Trying camera index {idx}...")
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[INFO] Successfully connected to camera {idx}")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                return cap
            else:
                cap.release()
        else:
            cap.release()
    print("[ERROR] No camera found. Please check your webcam/CCTV connection.")
    return None

cap = initialize_camera()
if cap is None:
    exit(1)

'''
Liveness detector initialization:
This sets up the system that checks if a detected face is real or fake.
'''
liveness_detector = None
if ENABLE_LIVENESS_DETECTION:
    try:
        liveness_detector = LivenessDetector(model_path="models/modelrgb.onnx", threshold=LIVENESS_THRESHOLD)
        print(f"[INFO] Liveness detector initialized with threshold: {LIVENESS_THRESHOLD}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize liveness detector: {e}")
        ENABLE_LIVENESS_DETECTION = False

'''
Data tracking variables
'''
last_logged = {}  # Dictionary to keep track of last attendance log time for each person

'''
Face detection function using YuNet:
Returns list of [startX, startY, endX, endY] boxes.
'''
def detect_faces_yunet(frame):
    if yunet is None:
        return []
    try:
        h, w = frame.shape[:2]
        yunet.setInputSize((w, h))
        retval, faces = yunet.detect(frame)
        boxes = []
        if faces is not None and len(faces) > 0:
            for face in faces:
                x, y, w_box, h_box, score = face[:5]
                if score >= 0.9:
                    startX, startY = max(0, int(x)), max(0, int(y))
                    endX = min(w - 1, int(x + w_box))
                    endY = min(h - 1, int(y + h_box))
                    if endX > startX and endY > startY:  # Only valid boxes
                        boxes.append([startX, startY, endX, endY])
        return boxes
    except Exception as e:
        print(f"[ERROR] YuNet face detection failed: {e}")
        return []

'''
Liveness detection function:
KEY: spoof_utils.preprocess() converts BGR→RGB internally,
so we must pass the BGR face crop (from original frame), NOT the rgb crop.
Passing RGB would cause double-swap → wrong colors → always SPOOF.
'''
def perform_liveness_detection(face_img_bgr):
    if not ENABLE_LIVENESS_DETECTION or liveness_detector is None:
        return False
    try:
        if face_img_bgr is None or face_img_bgr.size == 0:
            return False
        h, w = face_img_bgr.shape[:2]
        if h < 32 or w < 32:
            return False
        is_spoof = liveness_detector.is_spoof(face_img_bgr)  # Pass BGR; model converts internally
        if DEBUG_LIVENESS:
            try:
                confidence = liveness_detector.get_confidence(face_img_bgr)
                print(f"[DEBUG] Liveness: {confidence:.3f} → {'SPOOF' if is_spoof else 'REAL'}")
            except Exception:
                print(f"[DEBUG] Liveness result: {'SPOOF' if is_spoof else 'REAL'}")
        return is_spoof
    except Exception as e:
        if DEBUG_LIVENESS:
            print(f"[ERROR] Liveness failed: {e}, treating as REAL")
        return False  # Fail open on error

'''
Main program loop
'''
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame from camera.")
        break

    frame_count += 1

    # BUG FIX: Convert to RGB once upfront. All face operations use the rgb copy.
    # The liveness model expects RGB, and face_recognition also expects RGB.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes = detect_faces_yunet(frame)

    # Show frame counter in top-left at all times
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    if not boxes:
        cv2.putText(frame, "No faces detected", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)
    else:
        # face_recognition expects (top, right, bottom, left) and FULL frame must be in bounds
        fh, fw = rgb.shape[:2]
        face_locations = []
        for (startX, startY, endX, endY) in boxes:
            top = max(0, startY)
            right = min(fw, endX)
            bottom = min(fh, endY)
            left = max(0, startX)
            face_locations.append((top, right, bottom, left))

        try:
            face_encodings_list = face_recognition.face_encodings(rgb, face_locations)
        except Exception as e:
            print(f"[ERROR] Face encoding failed: {e}")
            face_encodings_list = []

        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H:%M:%S')

        for i, (startX, startY, endX, endY) in enumerate(boxes):
            name = "Unknown"
            confidence_text = ""
            box_color = (0, 0, 255)  # Default red = unknown

            # Pass BGR face crop to liveness (spoof_utils converts BGR→RGB internally)
            face_img_bgr = frame[startY:endY, startX:endX]
            is_spoof = perform_liveness_detection(face_img_bgr)

            if is_spoof:
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, "SPOOF", (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                continue  # Skip this face

            # Try to recognize the face
            if i < len(face_encodings_list) and len(data["encodings"]) > 0:
                encoding = face_encodings_list[i]
                distances = face_recognition.face_distance(data["encodings"], encoding)
                if len(distances) > 0:
                    min_distance = float(np.min(distances))
                    best_match_index = int(np.argmin(distances))
                    if min_distance < MIN_CONFIDENCE:
                        name = data["names"][best_match_index]
                        confidence_text = f" ({min_distance:.2f})"
                        box_color = (0, 255, 0)  # Green: recognized
                        print(f"[INFO] Recognized: {name} (Distance: {min_distance:.2f})")
                        # Log attendance with throttle
                        if name not in last_logged or (now - last_logged[name]).total_seconds() > RE_LOG_GAP:
                            try:
                                log_attendance(name, date_str, time_str)
                                last_logged[name] = now
                            except Exception as e:
                                print(f"[ERROR] Attendance logging failed: {e}")
                    else:
                        box_color = (0, 255, 255)  # Yellow: low confidence
                        confidence_text = f" ({min_distance:.2f})"
            elif len(data["encodings"]) == 0:
                box_color = (128, 128, 128)  # Grey: no encodings loaded

            cv2.rectangle(frame, (startX, startY), (endX, endY), box_color, 2)
            label = f"{name}{confidence_text}"
            cv2.putText(frame, label, (startX, endY + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Status bar at bottom
    liveness_status = f"Liveness: {'ON' if ENABLE_LIVENESS_DETECTION else 'OFF'}"
    liveness_color = (0, 255, 0) if ENABLE_LIVENESS_DETECTION else (0, 0, 255)
    cv2.putText(frame, liveness_status, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, liveness_color, 1)
    cv2.putText(frame, "Press 'q' to quit | 'l' to toggle liveness", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imshow("Face Recognition Attendance", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("[INFO] Quitting attendance tracking.")
        break
    elif key == ord('l'):
        ENABLE_LIVENESS_DETECTION = not ENABLE_LIVENESS_DETECTION
        print(f"[INFO] Liveness detection {'ENABLED' if ENABLE_LIVENESS_DETECTION else 'DISABLED'}")

cap.release()
cv2.destroyAllWindows()
print("[INFO] Application closed successfully.")