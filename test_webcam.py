'''
OpenCV (Open Source Computer Vision Library) is a powerful library used for:
- Image and video processing
- Face detection & recognition
- Object tracking, motion detection
- Computer vision in AI/ML projects

This script is a simple webcam testing tool that helps you check if your camera is working properly.
It's like a "camera test" before using the main attendance system.
'''
import cv2  # Import the OpenCV library for working with video and images
import sys  # Import the sys module to access command-line arguments
def main(camera_index=0):
    '''
    This function starts the webcam using OpenCV.
    It takes an optional argument (camera_index), which lets you select which webcam to use (default is 0).
    
    Camera indices explained:
    - 0: Usually the built-in laptop webcam or first USB camera
    - 1: Second camera if multiple cameras are connected
    - 2, 3, etc.: Additional cameras
    '''
    cap = cv2.VideoCapture(camera_index)  # Create a VideoCapture object to access the webcam
    if not cap.isOpened():  # Check if the webcam is successfully opened
        print("❌ Error: Could not open webcam.")
        return  # Stop the program if the webcam cannot be accessed
    print("✅ Webcam started successfully. Press 'q' in the window or Ctrl+C in terminal to exit.")
    try:
        '''
        Main video loop:
        This loop continuously captures frames from the webcam and displays them.
        It runs until the user decides to quit.
        '''
        while True:  # Keep running until user quits or an error occurs
            ret, frame = cap.read()  # Read one frame from the webcam. ret = True if frame was captured(or returned) successfully, False if failed. frame = the actual image/video frame from the webcam
            if not ret:  # If frame is not read successfully
                print("❌ Failed to grab frame.")
                break  # Exit the loop
            '''
            Add instruction text to the video frame:
            This puts text on the video to show the user how to exit the program.
            '''
            # Add instruction text on the video frame
            cv2.putText(frame, "Press 'q' to quit", (10, 30),  # Position: (10,30) from top-left corner
                        cv2.FONT_HERSHEY_SIMPLEX, 1,          # Font type and size (1 = medium size)
                        (0, 255, 0), 2)                        # Green color (BGR format), thickness = 2 pixels. (0, 255, 0) → Green; (255, 0, 0) → Blue; (0, 0, 255) → Red
            
            cv2.imshow("Webcam Test", frame)  # Display the name "Webcam Test" on the pop up window at the top of the frame i.e., on title bar 
            '''
            Check for key press:
            cv2.waitKey(1) waits for 1 millisecond for a key press. If not, continue looping (used in real-time video/live camera). And returns a 32-bit integer (usually platform-dependent), if key is pressed.
            if we had written this cv2.waitKey(0) & 0xFF == ord('q'). Then this means: The program will pause and wait forever until any key is pressed. Only then it checks if the key was 'q'.
            But only the last 8 bits (1 byte) represent the actual key code. So, & 0xFF is used to extract just those last 8 bits.
            ord('q') gets the ASCII code of the 'q' character.
            '''
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Check if the 'q' key is pressed
                break  # Exit the loop
    except KeyboardInterrupt:
        '''
        This block runs when the user presses Ctrl+C in the terminal.
        It's a graceful way to exit the loop when interrupting the program.
        KeyboardInterrupt is a special exception that Python raises when Ctrl+C is pressed.
        '''
        print("\n⏹️ Exiting on keyboard interrupt.")
    finally:
        '''
        Cleanup section:
        This block always runs at the end, even if there was an error.
        It ensures the webcam is properly closed and windows are destroyed.
        '''
        # These commands release the webcam and close the window when the loop ends
        cap.release()  # Stop the webcam and free up the camera resource
        cv2.destroyAllWindows()  # Close all OpenCV windows that were opened
'''
Main program execution:
This section only runs when the script is executed directly (not when imported as a module).
This block handles command-line arguments and starts the main function.
'''
if __name__ == "__main__":
    '''
    This checks if the script is being run directly (not run when imported).
    It optionally accepts a camera index from the command line.
    q
    Example usage: If we right any noe of the below line in cmd and then run this script, then what will happen?
    python test_webcam.py     # means we are not specifying any index so, it Uses camera index 0 (default). Then the function, def main(camera_index=0) will take this default value.
    python test_webcam.py 1   # Uses camera index 1
    python test_webcam.py 2   # Uses camera index 2
    '''
    index = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # Use first argument if given, else default to 0. sys.argv[1] is the first command-line argument after the script name. If no argument is provided, use camera index 0
    main(index)  # Call the main function with the selected camera index
    '''
    1. sys.argv: It’s a list that stores the command-line arguments.
                 Example:
                If you run: python test_webcam.py 2. Then: sys.argv[0] = 'test_webcam.py' (Script name), and, sys.argv[1] = '2' (First argument (as string))
    2. len(sys.argv) > 1: Checks if any argument is given after the script name.
    3. int(sys.argv[1]): Converts the first argument ('2' in our example) into an integer.
    4. else 0: If no argument is provided, use 0 as default.

    "If the user gave a camera index (like 1 or 2) when running the script, use that. If not, use 0 by default."
    '''