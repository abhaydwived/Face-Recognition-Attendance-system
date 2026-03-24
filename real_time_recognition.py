'''
This script performs real-time face recognition using a webcam or CCTV.
It loads known face encodings, detects faces in each video frame, and recognizes them.
This is for RECOGNITION ONLY - no attendance logging.

HOW IT WORKS:
1. Opens your camera (webcam or CCTV)
2. Looks at each video frame to find faces
3. Compares found faces with known people in the database
4. Displays names above recognized faces in real-time
5. Shows confidence scores for recognition accuracy
'''
import cv2  # Import OpenCV for video capture and image processing - this handles camera and video
import face_recognition  # Import face_recognition for face detection and recognition - this finds and matches faces
import pickle  # Import pickle to load saved face encodings - this reads the saved face data
import os  # Import os for file system operations - this checks if files exist
from datetime import datetime  # Import datetime for handling date and time - this gets current time

MIN_CONFIDENCE = 0.35  # Threshold for face recognition match (lower = stricter)
YUNET_MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"  # Path to the YuNet face detection model
'''
Load known face encodings from a file.
If the file is missing, the script will exit with an error.
'''
try:
    with open('encodings/face_encodings.pkl', 'rb') as f:  # Open the encodings file in binary read mode
        data = pickle.load(f)  # Load the pickled data (should contain 'encodings' and 'names')
    print(f"[INFO] Loaded {len(data['encodings'])} face encodings.")  # Print number of encodings loaded
    print(f"[INFO] Known names: {set(data['names'])}")  # Print the set of known names
except FileNotFoundError:  # If the file is not found
    print("[ERROR] Face encodings not found! Please run encode_faces.py first.")  # Print error message
    exit(1)  # Exit the script
'''
This checks if the AI model file for face detection is available.
The YuNet model is what actually finds faces in the camera image.
'''
if not os.path.isfile(YUNET_MODEL_PATH):  # If the model file is not found
    print(f"[ERROR] YuNet model not found at {YUNET_MODEL_PATH}")  # Print error
    exit(1)  # Exit script
'''
Try to connect to a camera (webcam or CCTV). Try indices 0, 1, 2.
The first camera that works will be used for video capture.
'''
cap = None  # Initialize camera variable
for camera_index in [0]:  # Try camera indices 0, 1, and 2
    cap = cv2.VideoCapture(camera_index)  # Attempt to open the camera
    if cap.isOpened():  # If camera opens successfully
        print(f"[INFO] Camera connected at index {camera_index}")  # Print which index worked
        break  # Stop trying other cameras
    cap.release()  # Release the camera if not successful
if not cap or not cap.isOpened():  # If no camera was opened
    print("[ERROR] Could not connect to camera!")  # Print error message
    exit(1)  # Exit script
# Read one frame to get the size for YuNet
'''
This takes one picture from the camera to find out its size (width and height).
The AI model needs to know the image size to work properly.
'''
ret, frame = cap.read()  # Read a single frame from the camera
if not ret:  # If frame was not read successfully
    print("[ERROR] Failed to capture frame from camera")  # Print error
    cap.release()  # Release the camera
    exit(1)  # Exit script
h, w, _ = frame.shape  # Get the height, width, and channels of the frame
'''
Initialize the YuNet face detector using the model file and frame size.
YuNet is a fast and accurate face detection model.

WHAT THIS DOES:
- Sets up the AI system that will find faces in camera images
- Configures it with the right image size and sensitivity settings
- score_threshold: How confident the AI needs to be that it found a face (0.9 = 90% sure)
- nms_threshold: Prevents finding the same face multiple times
- top_k: Maximum number of faces it can find at once
'''
detector = cv2.FaceDetectorYN_create(
    YUNET_MODEL_PATH,  # Path to YuNet model
    "",  # No config file
    (w, h),  # Input size (width, height)
    score_threshold=0.9,  # Only accept detections with high confidence
    nms_threshold=0.3,  # Non-maximum suppression threshold
    top_k=100  # Maximum number of faces to detect
)

print("[INFO] Face recognition started. Press 'q' to quit.")  # Notify user that recognition has started
'''
Detect faces in a frame using YuNet and return bounding boxes in the format required by face_recognition.
WHAT THIS FUNCTION DOES:
- Takes a camera image and finds all the faces in it
- Returns the coordinates of where each face is located
- These coordinates are like invisible rectangles drawn around each face
'''
def detect_faces_yunet(frame, detector):  # Define the function for face detection
    h, w = frame.shape[:2]  # Get frame height and width
    detector.setInputSize((w, h))  # Set detector input size
    retval, faces = detector.detect(frame)  # Detect faces
    boxes = []  # List to store bounding boxes
    if faces is not None and len(faces) > 0:  # If faces are detected
        for face in faces:  # Loop through each detected face
            x, y, w_box, h_box, score = face[:5]  # Get box coordinates and score
            if score >= 0.9:  # Only use boxes with high confidence
                left = int(x)  # Calculate left coordinate
                top = int(y)  # Calculate top coordinate
                right = int(x + w_box)  # Calculate right coordinate
                bottom = int(y + h_box)  # Calculate bottom coordinate
                boxes.append((top, right, bottom, left))  # Append box in correct format
    return boxes  # Return all detected boxes
'''
Main loop: Reads frames from the camera, detects and recognizes faces, and displays results.
Press 'q' to quit the application.
THIS IS THE MAIN PROGRAM THAT RUNS CONTINUOUSLY:
1. Takes pictures from the camera continuously
2. Looks for faces in each picture
3. Tries to recognize who each face belongs to
4. Shows the live camera feed with names above faces
5. Displays confidence scores for recognition accuracy
6. Continues until you press 'q' to quit
'''
try:
    while True:  # Loop forever until user quits
        ret, frame = cap.read()  # Read a frame from the camera
        if not ret:  # If frame not read successfully
            print("[ERROR] Failed to capture frame from camera")  # Print error
            break  # Exit loop
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
        boxes = detect_faces_yunet(frame, detector)  # Detect faces
        face_encodings = face_recognition.face_encodings(rgb, boxes)  # Get face encodings
        current_time = datetime.now().strftime('%H:%M:%S')  # Get current time for display
        # Process each detected face
        for encoding, box in zip(face_encodings, boxes):  # Loop through each face
            distances = face_recognition.face_distance(data["encodings"], encoding)  # Calculate distances to known faces
            min_distance = min(distances)  # Find closest match
            best_match_index = distances.argmin()  # Index of closest match
            name = "Unknown"  # Default to unknown
            confidence_text = f"({min_distance:.2f})"  # Show confidence score
            '''
            Check if the face matches anyone we know.
            If the distance is small enough, it's probably the same person.
            '''
            if min_distance < MIN_CONFIDENCE:  # If match is close enough
                name = data["names"][best_match_index]  # Get matched name
                print(f"[✅] Recognized: {name} (Distance: {min_distance:.2f})")  # Print match info
            else:  # If not recognized
                print(f"[INFO] Unknown person detected (Distance: {min_distance:.2f})")  # Print info
            '''
            Draw rectangles around faces and write names above them.
            Green rectangle = known person, Red rectangle = unknown person
            '''
            # Draw face box and name
            top, right, bottom, left = box  # Unpack box coordinates
            # Choose color based on recognition
            if name != "Unknown":
                color = (0, 255, 0)  # Green for known person
                text_color = (255, 255, 255)  # White text
            else:
                color = (0, 0, 255)  # Red for unknown person
                text_color = (255, 255, 255)  # White text
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)  # Draw rectangle around face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED) # Draw filled rectangle for name background
            # Write name and confidence
            label = f"{name.title()} {confidence_text}"
            cv2.putText(frame, label, (left + 5, bottom - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        # Add information overlay to the screen
        cv2.putText(frame, f"Time: {current_time}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Faces detected: {len(face_encodings)}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Face Recognition", frame)  # Show the frame
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Check if user wants to quit
            break  # Exit loop
except Exception as e:  # If any error occurs
    print(f"[ERROR] An error occurred: {str(e)}")  # Print error
finally:
    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all OpenCV windows
    print("[INFO] Camera released and windows closed")  # Print info