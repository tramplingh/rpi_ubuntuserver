import numpy as np
import cv2
import cv2.aruco as aruco
import time
import sys

# ================= SETTINGS =================
ID_TO_FIND = 72
MARKER_SIZE = 0.2035   # meters 
CALIB_PATH = "camera_01/"
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
SMOOTHING_ALPHA = 0.7
# ============================================

# Load calibration
camera_matrix = np.loadtxt(CALIB_PATH + "cameraMatrix.txt", delimiter=',')
camera_distortion = np.loadtxt(CALIB_PATH + "cameraDistortion.txt", delimiter=',')

# ArUco dictionary (must match your marker)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# Open camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

if not cap.isOpened():
    print("❌ Could not open camera")
    sys.exit()

print("🚀 Precision Landing Pose Estimation Started")
print("Press CTRL+C to stop\n")

prev_tvec = None
prev_time = time.time()

print("Camera matrix:\n", camera_matrix)
print("Distortion:\n", camera_distortion)
print("Resolution set:", CAMERA_WIDTH, "x", CAMERA_HEIGHT)
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, rejected = detector.detectMarkers(gray)
        print(ids) 
        if ids is not None:
            for i in range(len(ids)):
                if ids[i][0] == ID_TO_FIND:

                    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                        corners[i],
                        MARKER_SIZE,
                        camera_matrix,
                        camera_distortion
                    )

                    rvec = rvec[0][0]
                    tvec = tvec[0][0]

                    # ===== Smooth Translation (reduce jitter) =====
                    if prev_tvec is None:
                        prev_tvec = tvec
                    else:
                        tvec = SMOOTHING_ALPHA * prev_tvec + (1 - SMOOTHING_ALPHA) * tvec
                        prev_tvec = tvec

                    # ===== Landing Errors =====
                    x_error = tvec[0]   # left/right
                    y_error = tvec[1]   # forward/back
                    z_height = tvec[2]  # height
                    distance = np.linalg.norm(tvec)

                    # ===== Rotation (Roll, Pitch, Yaw) =====
                    R, _ = cv2.Rodrigues(rvec)
                    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)

                    roll  = np.arctan2(R[2,1], R[2,2])
                    pitch = np.arctan2(-R[2,0], sy)
                    yaw   = np.arctan2(R[1,0], R[0,0])

                    # Convert to degrees
                    roll  = np.degrees(roll)
                    pitch = np.degrees(pitch)
                    yaw   = np.degrees(yaw)

                    # ===== FPS Calculation =====
                    current_time = time.time()
                    fps = 1.0 / (current_time - prev_time)
                    prev_time = current_time

                    # ===== Output =====
                    print(
                        f"\rFPS: {fps:5.1f} | "
                        f"X: {x_error:+.3f} m  "
                        f"Y: {y_error:+.3f} m  "
                        f"Z: {z_height:.3f} m  "
                        f"Dist: {distance:.3f} m  | "
                        f"Yaw: {yaw:+.1f}°",
                        end=""
                    )

except KeyboardInterrupt:
    print("\n\nStopped by user.")

cap.release()
