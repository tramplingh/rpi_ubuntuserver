import cv2
import cv2.aruco as aruco

def main():
    # 1. Initialize the USB camera (Video device 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open the Logitech camera.")
        return

    print("Camera initialized. Capturing frame...")
    
    # Allow the camera a moment to auto-focus and adjust white balance
    for _ in range(5):
        cap.read()

    # Capture the actual frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab a frame.")
        cap.release()
        return

    # 2. Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3. Set up the ArUco detector (Modern OpenCV 4.7+ Syntax)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # 4. Detect markers
    corners, ids, rejected = detector.detectMarkers(gray)

    # 5. Process results
    if ids is not None:
        print(f"Success! Detected ArUco marker IDs: {ids.flatten()}")
        
        # Draw outlines and IDs on the original image
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        # Save the image to the Pi's storage
        output_filename = "aruco_output.jpg"
        cv2.imwrite(output_filename, frame)
        print(f"Saved annotated image as '{output_filename}'")
    else:
        print("No ArUco markers found in the camera's view.")

    # Clean up
    cap.release()

if __name__ == "__main__":
    main()