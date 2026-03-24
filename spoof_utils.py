import cv2  # OpenCV library for computer vision tasks like image processing
import numpy as np  # NumPy for numerical operations and array handling
import onnxruntime as ort  # used to run ONNX (Open Neural Network Exchange) models in Python using the ONNX Runtime library.
import os  # Operating system interface for file and directory operations
class LivenessDetector:
    '''
    This class is responsible for detecting if a face in an image is real (from a live person) 
    or fake (from a photo/video). This prevents people from fooling the system by showing 
    a picture of someone else to the camera.
    '''
    def __init__(self, model_path, threshold=0.5):
        self.threshold = threshold # Store the confidence threshold (0.5 means 50% confidence needed to consider face as real)
        self.model_loaded = False # Flag to track if the AI model loaded successfully
        if not os.path.exists(model_path): # Check if the AI model file exists at the specified path
            print(f"[WARNING] ONNX model not found at {model_path}")
            print("[INFO] Liveness detection will be disabled")
            return  # Exit early if model file doesn't exist   
        try:
            '''
            Try to load the ONNX model (a type of AI model format)
            This model has been trained to distinguish between real faces and fake ones
            '''
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider']) # Create an inference(prediction) session with the model, using CPU for processing
            # There’s no need to access any other index like [1], [2], etc., because those don’t exist unless the model is unusually complex (e.g. multi-input models).
            self.input_name = self.session.get_inputs()[0].name # Get the name of the input layer of the neural network
            self.input_shape = self.session.get_inputs()[0].shape # Get the expected input shape (dimensions) that the model expects
            self.model_loaded = True # Mark that model loaded successfully
            print(f"[INFO] ONNX model loaded successfully. Input shape: {self.input_shape}")
            print(f"[INFO] Input name: {self.input_name}")  # Print success messages with model details
        except Exception as e:  # If loading fails, print error and disable liveness detection
            print(f"[ERROR] Failed to load ONNX model: {e}")
            print("[INFO] Liveness detection will be disabled")
    def is_model_available(self):
        '''
        Simple function to check if the liveness detection model is ready to use
        Returns True if model is loaded, False otherwise
        '''
        return self.model_loaded
    def preprocess(self, face_img):
        '''
        This function prepares a face image for the AI model to analyze.
        The image needs to be in exactly the right format for the AI to understand it.
        '''
        try:
            if face_img is None or face_img.size == 0: # Check if the face image is valid (not empty or corrupted)
                raise ValueError("Invalid face image")
            expected_size = self.input_shape[2] if len(self.input_shape) >= 3 else 64  # Get the expected image size from the model (usually 64x64 pixels)
            img = cv2.resize(face_img, (expected_size, expected_size), interpolation=cv2.INTER_LINEAR)  # Resize the face image to match what the model expects
            if len(img.shape) == 3 and img.shape[2] == 3: # len(img.shape) == 3 → The image has 3 dimensions: Height (H), Width (W), Channels (C)  and , img.shape[2] == 3 → The number of channels is 3 (which means it's a color image).
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # Convert color format from BGR (Blue-Green-Red) to RGB (Red-Green-Blue)
            img = img.astype(np.float32) / 255.0  # Normalize pixel values to be between 0 and 1 (instead of 0-255)
            img = np.transpose(img, (2, 0, 1))  # Rearrange dimensions from (height, width, channels) to (channels, height, width)
            img = np.expand_dims(img, axis=0)  # Add a batch dimension (needed for neural network processing)
            if img.shape != tuple(self.input_shape):   # Check if the final shape matches what the model expects
                print(f"[WARNING] Shape mismatch. Expected: {self.input_shape}, Got: {img.shape}")
            return img  # Return the processed image
        except Exception as e:
            print(f"[ERROR] Preprocessing failed: {e}")
            return None  # Return None if preprocessing fails
    def is_spoof(self, face_img):
        '''
        This is the main function that determines if a face is real or fake.
        It analyzes the face image and returns True if it's fake (spoof) or False if it's real.
        '''
        if not self.model_loaded:  # If model isn't loaded, skip detection and assume face is real
            print("[INFO] Liveness model not available, skipping spoof detection")
            return False
        try:
            input_tensor = self.preprocess(face_img)  # Prepare the face image for the AI model
            if input_tensor is None:   # If preprocessing failed, assume it's a spoof for security
                print("[WARNING] Preprocessing failed, treating as spoof")
                return True
            outputs = self.session.run(None, {self.input_name: input_tensor})  # Run the model on the input image and get the output predictions typically contains bounding boxes, class scores, etc., depending on the model type
            if len(outputs) > 0:
                output = outputs[0]  # Get the first output
                '''
                The model output can be in different formats, so we need to handle each case:
                - Case 1: Two values [spoof_score, live_score]
                - Case 2: One value [live_score]
                - Case 3: Array with multiple values
                '''
                # Case 1: Output has 2 values (spoof probability and live probability)
                if len(output.shape) == 2 and output.shape[1] == 2:
                    live_score = output[0][0]  # Probability that face is real
                    spoof_score = output[0][1]  # Probability that face is fake
                # Case 2: Output has 1 value (live probability only)
                elif len(output.shape) == 2 and output.shape[1] == 1:
                    live_score = output[0][0]  # Probability that face is real
                    spoof_score = 1.0 - live_score  # Calculate spoof probability
                # Case 3: 1D array output
                elif len(output.shape) == 1:
                    if len(output) >= 2:
                        live_score = output[0]  # Second value is live score
                        spoof_score = output[1]  # First value is spoof score
                    else:
                        live_score = output[0]  # Only one value available
                        spoof_score = 1.0 - live_score  # Calculate opposite
                else:  # Unexpected output format - try to handle it gracefully
                    print(f"[WARNING] Unexpected output shape: {output.shape}")
                    live_score = output.flatten()[0] if output.size > 0 else 0.0
                    spoof_score = 1.0 - live_score
                is_spoof = live_score < self.threshold # Determine if face is fake: if live_score is below threshold, it's a spoof
                # Print debug information for troubleshooting
                print(f"[DEBUG] Live score: {live_score:.3f}, Spoof score: {spoof_score:.3f}, Threshold: {self.threshold:.3f}")
                print(f"[DEBUG] Result: {'SPOOF' if is_spoof else 'REAL'}")
                return is_spoof  # Return True if fake, False if real
            else:
                print("[ERROR] No output from model")
                return True  # If no output, assume it's fake for security
        except Exception as e:
            print(f"[ERROR] Liveness detection failed: {e}")
            return True  # If error occurs, assume it's fake for security
    def get_confidence(self, face_img):
        '''
        This function returns how confident the model is that the face is real.
        Returns a number between 0 and 1, where 1 means 100% confident it's real.
        '''
        if not self.model_loaded:   # If model isn't available, return maximum confidence (assume real)
            return 1.0
        try:
            input_tensor = self.preprocess(face_img)  # Prepare the image for the model
            if input_tensor is None:  # If preprocessing failed, return 0 confidence
                return 0.0
            outputs = self.session.run(None, {self.input_name: input_tensor})  # Run the model to get predictions
            if len(outputs) > 0:
                output = outputs[0]
                if len(output.shape) == 2 and output.shape[1] == 2:  # Handle different output formats and return the live score
                    return float(output[0][0])  # Return live probability
                elif len(output.shape) == 2 and output.shape[1] == 1:
                    return float(output[0][0])  # Return single probability
                else:
                    return float(output.flatten()[0]) if output.size > 0 else 0.0
            return 0.0  # Return 0 if no output
        except Exception as e:
            print(f"[ERROR] Getting confidence failed: {e}")
            return 0.0  # Return 0 on error
class FaceValidator:
    '''
    This class is responsible for validating uploaded photos to ensure they contain exactly one face.
    It's like a quality checker that makes sure uploaded photos are good enough for the system to use.
    It uses YuNet (a face detection model) to find faces in images.
    '''
    YUNET_MODEL_PATH = os.path.join("models", "face_detection_yunet_2023mar.onnx")  # Path to the YuNet face detection model file
    NETWORK_INPUT_SIZE = (320, 320)    # Standard input size for the YuNet model (320x320 pixels)
    _detector = None  # Class variable to store the detector instance (shared across all instances)
    @classmethod 
    def _load_detector(cls):
        '''
        This function loads the YuNet face detection model.
        It's marked as @classmethod which means it belongs to the class, not individual instances.
        '''
        if cls._detector is None:    # Only load the detector once (if not already loaded)
            # Check if the model file exists
            if not os.path.isfile(cls.YUNET_MODEL_PATH):
                raise FileNotFoundError(f"YuNet model not found at {cls.YUNET_MODEL_PATH}")
            # Create the face detector using OpenCV's YuNet implementation
            cls._detector = cv2.FaceDetectorYN_create(
                cls.YUNET_MODEL_PATH,  # Path to model file
                "",  # Empty config file path
                cls.NETWORK_INPUT_SIZE,  # Input size for the network
                score_threshold=0.9  # Confidence threshold (90% confidence needed)
            )
    @classmethod
    def validate_uploaded_image(cls, image_path):
        '''
        This function checks if an uploaded image is suitable for face recognition.
        It returns True/False and a message explaining the result.
        Requirements:
        - Image must be readable
        - Exactly one face must be detected
        - Face must be clear enough (90% confidence)
        '''
        try:
            cls._load_detector()  # Load the face detector
            img = cv2.imread(image_path)  # Read the image from the file path
            if img is None: # Check if image was loaded successfully
                return False, "Could not load image file"
            h, w = img.shape[:2]  # Get image dimensions (height and width)
            cls._detector.setInputSize((w, h))   # Set the input size for the detector to match the image size
            retval, faces = cls._detector.detect(img)   # Detect faces in the image
            if faces is None or len(faces) == 0:   # Check the results and return appropriate message
                return False, "No face detected in the image"
            elif len(faces) > 1:
                return False, "Multiple faces detected. Please upload image with single face"
            else:
                return True, "Face detected successfully"
        except Exception as e:
            return False, f"Error validating image: {str(e)}"  # Return error message if something goes wrong
    @classmethod
    def extract_face_from_image(cls, image_path, output_size=(128, 128)):
        '''
        This function extracts just the face part from an image and resizes it.
        It's like cropping a photo to show only the face, then making it a standard size.
        Returns the cropped and resized face image, or None if no face is found.
        '''
        try:
            cls._load_detector()  # Load the face detector
            img = cv2.imread(image_path) # Read the image from file 
            if img is None: # Check if image was loaded successfully
                return None
            h, w = img.shape[:2]  # Get image dimensions
            cls._detector.setInputSize((w, h)) # Set detector input size to match image
            retval, faces = cls._detector.detect(img) # Detect faces in the image
            if faces is not None and len(faces) > 0:  # If at least one face is found, extract the first one
                x, y, w_box, h_box, score = faces[0][:5]  # Get the coordinates and dimensions of the first face
                x, y, w_box, h_box = map(int, [x, y, w_box, h_box])  # Convert coordinates to integers
                face = img[y:y+h_box, x:x+w_box] # Crop the face from the original image
                face_resized = cv2.resize(face, output_size) # Resize the face to the desired output size
                return face_resized  # Return the processed face
            return None  # Return None if no face found
        except Exception as e:
            print(f"[ERROR] Error extracting face: {e}")
            return None  # Return None if error occurs