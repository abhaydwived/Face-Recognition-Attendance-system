import cv2  # OpenCV for camera access and image processing
import os   # For working with file and directory paths
import time  # Used for delays between instructions
# import pyttsx3  # Text-to-speech engine for giving voice instructions
import shutil  # For deleting directories and their contents

YUNET_MODEL_PATH = os.path.join("models", "face_detection_yunet_2023mar.onnx") # YuNet ONNX model path for face detection
POSES = [
    "face front towards camera",           # Pose 1: Looking straight ahead
    "turn face slightly to the left",      # Pose 2: Head turned left
    "turn face slightly to the right",     # Pose 3: Head turned right
    "look up slightly",                    # Pose 4: Looking up
    "look down slightly"                   # Pose 5: Looking down
] # List of 5 different face poses we want the user to show
# def speak(text):
#     '''
#     This function converts text to speech using the computer's voice.
#     The function both speaks the text out loud AND prints it on screen,
#     so the user can hear AND see the instructions.
#     '''
#     print(f"[Instruction] {text}")  # Print the instruction text on screen
#     engine.say(text)  # Tell the text-to-speech engine what to say
#     engine.runAndWait()  # Wait for the speech to finish before continuing
def capture_faces(person_name, camera_index=0):
    '''
    This is the main function that captures 5 face photos of a person.
    Think of it like a photo booth session where:
    1. The computer tells you how to pose
    2. You adjust your face position
    3. Press ENTER when ready
    4. The computer saves a cropped picture of just your face
    5. Repeat for 5 different poses
    All photos are saved in a folder named after the person.
    '''
    script_dir = os.path.dirname(os.path.abspath(__file__)) # Get the directory where this script is located
    dataset_path = os.path.join(script_dir, "dataset", person_name) # Create the path where photos will be saved: dataset/person_name/
    if os.path.exists(dataset_path): # If folder for this person already exists, delete it to avoid mixing old and new photos
        shutil.rmtree(dataset_path)  # Delete the entire folder and its contents
    os.makedirs(dataset_path, exist_ok=True) # Create a fresh, empty folder for this person's photos
    cap = cv2.VideoCapture(camera_index)  # camera_index=0 means the default camera
    if not cap.isOpened(): # Check if camera is working
        print("❌ Error: Webcam not accessible.")
        return  # Exit the function if camera doesn't work
    # Read one frame to get dimensions (width and height)
    ret, frame = cap.read()  # ret=True if successful, frame=the actual image
    if not ret:
        print("❌ Failed to read from camera.")
        cap.release()  # Close the camera
        return  # Exit the function
    # Get frame dimensions
    h, w, _ = frame.shape  # h=height, w=width, _=color channels (ignored)
    # Initialize YuNet face detector
    '''
    YuNet is an AI model that can find faces in images.
    Think of it as a smart system that can point to where faces are located.
    '''
    detector = cv2.FaceDetectorYN_create(
        YUNET_MODEL_PATH,     # Path to the AI model file
        "",                   # No additional config file needed. This is usually used for extra settings if required by some models.
        (w, h),               # Input size = webcam frame size
        score_threshold=0.9,  # Minimum confidence (90%) to accept a face detection
        nms_threshold=0.3,    # If multiple boxes overlap too much (more than 30%), it keeps only the best one. For removing duplicate/overlapping face boxes
        top_k=50              # Maximum number of faces to detect in one image
    )  # FaceDetectorYN_create is a function inside cv2 that initializes the face detector with the specified model and settings.
  #  speak(f"Hello {person_name}, we will now capture 5 cropped face photos.") # Welcome message - greet the person and explain what will happen
    for i, pose in enumerate(POSES, start=1):  # Loop through each of the 5 poses. start=1 means we start counting from 1 instead of 0 (enumerate by default starts from 0)
        '''
        This loop runs 5 times, once for each pose. For each pose, it:
        1. Tells the user what pose to do
        2. Waits for them to adjust their position
        3. Shows live camera feed with face detection
        4. Waits for user to press ENTER
        5. Saves the cropped face image
        '''
      #  speak(f"Please {pose}")   # Tell the user what pose to do (e.g., "Please face front towards camera")
        time.sleep(2) # Give the user 2 seconds to adjust their position
        while True: # Keep showing camera feed until user presses ENTER
            ret, frame = cap.read() # Read the current frame from the camera
            if not ret:
                print("❌ Failed to capture frame.")
                continue  # Try again if frame reading failed
            if frame.shape[0] != h or frame.shape[1] != w: # If webcam resolution changes, update the detector input size
                h, w, _ = frame.shape  # Update stored dimensions
                detector.setInputSize((w, h))  # Tell detector about new size
            retval,faces = detector.detect(frame) # Use AI to detect faces in the current frame. retval means return value to check if face detection ran successfully.
            face_found = False # Flag to track if we found a face, initially set to False  that we do not found a face yet
            if faces is not None and len(faces) > 0: # Check if any faces were detected. faces is a list of face detections. Each face is usually like: [x, y, w, h, score, ...]
                '''
                If multiple faces are detected, we want to pick the best one.
                Each face detection comes with a confidence score.
                We sort by confidence and pick the highest scoring face.
                '''
                # Syntax: sorted(iterable, key=None, reverse=False)
                # Parameters: iterable: The sequence (like list, tuple, etc.) you want to sort. key (optional): A function that serves as a sorting key. It tells Python what to sort by. Default is False (ascending).
                faces = sorted(faces, key=lambda x: x[4], reverse=True) #  x[4] picks the 5th value i.e., confidence score in each face's data. Then, sort faces by confidence score (highest first means in descending order)
                # Get the coordinates of the best face
                x, y, w_box, h_box, score = faces[0][:5] # [0] is used to select the detected face ( at 0 is the face with highest confidence). [:5] means "take the first 5 values" from that detection which are [x, y, w_box, h_box, score, ...]
                # x, y = top-left corner of face box ; w_box, h_box = width and height of face box , score = confidence level
                x, y, w_box, h_box = map(int, [x, y, w_box, h_box]) # Convert coordinates to integers (required for drawing green box)
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2) # Draw a green rectangle around the detected face, where (0, 255, 0) = green color in BGR format and, 2 = thickness of the rectangle border
                face_found = True # Mark that we found a face
            cv2.imshow("Capture - Press ENTER to capture", frame) # Show the camera feed with face detection box
            key = cv2.waitKey(1)  # # Check if any key was pressed, wait 1 millisecond for a key press
            if key == 13: # If ENTER key was pressed (whose key code is 13)
                if not face_found: # Check if we detected a face
                  #  speak("No face detected. Please adjust your position.")
                    continue  # Go back to camera feed, don't save anything
                face_img = frame[y:y + h_box, x:x + w_box] # This extracts just the rectangular area containing the face
                img_path = os.path.join(dataset_path, f"{i}.jpg") # Create the file path where this photo will be saved. Example: dataset/john_doe/1.jpg, dataset/john_doe/2.jpg, etc.
                cv2.imwrite(img_path, face_img) # Save the face image to disk
                # speak(f"Captured photo {i}") # Confirm to user that photo was saved
                print(f"[Saved] {img_path}")
                break # Break out of the while loop to move to next pose
        time.sleep(1) # Small delay before starting the next pose
    # speak("All 5 face photos captured successfully.")
    cap.release()  # Stop using the camera
    cv2.destroyAllWindows()  # Close all OpenCV windows

if __name__ == "__main__":
    '''
    This block only runs when the script is executed directly (not imported as a module).
    It's like the "main" function that starts everything when you run the script.
    '''
    # engine = pyttsx3.init() # Initialize the text-to-speech engine to make the engine ready
    # engine.setProperty('rate', 160) # Set speaking speed to 160 words per minute (comfortable listening speed)
    person_name = input("Enter the name of the person: ").strip() # Ask the user to enter the person's name and .strip() removes any extra spaces before/after the name
    if person_name: # Check if a name was actually entered
        capture_faces(person_name) # Start the face capture process
    else:
        print("❌ Name cannot be empty.")  # Show error if no name was provided