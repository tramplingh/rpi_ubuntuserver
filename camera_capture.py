import cv2
import numpy as np
import os
import time

# ===== SETTINGS =====
save_folder = "camera_01"
image_width = 1280
image_height = 720
nRows = 9        # inner corners (must match calibration script)
nCols = 6
max_images = 20
capture_delay = 2  # seconds between valid captures
# ====================

os.makedirs(save_folder, exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)

if not cap.isOpened():
    print("❌ Could not open camera")
    exit()

print("Starting smart capture...")
print("Move the chessboard around.")
print(f"Will save {max_images} good images.\n")

img_counter = 0
last_capture_time = 0

while img_counter < max_images:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect chessboard
    found, corners = cv2.findChessboardCorners(gray, (nCols, nRows), None)

    if found:
        current_time = time.time()

        # Avoid saving too fast
        if current_time - last_capture_time > capture_delay:
            img_name = f"{save_folder}/calib_{img_counter:02d}.jpg"
            cv2.imwrite(img_name, frame)
            print(f"✅ Chessboard detected — Saved: {img_name}")

            img_counter += 1
            last_capture_time = current_time
        else:
            print("Chessboard detected... waiting before next capture")

    time.sleep(0.1)

cap.release()
print("\n🎯 Done collecting calibration images.")
