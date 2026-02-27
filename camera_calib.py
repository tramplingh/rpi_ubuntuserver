import cv2
import numpy as np
import glob
import os

# ============ SETTINGS ============
CHESSBOARD_SIZE = (9, 6)  # 10x7 squares = 9x6 inner corners
SQUARE_SIZE = 0.018       # Measure ONE square in meters! (e.g., 2.5cm = 0.025m)
RESOLUTION = (1280, 720)  # Must match your pose estimation code!
CALIBRATION_DIR = "camera_01/"
OUTPUT_DIR = "calibration_output/"  # Save debug images here
# ==================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Prepare 3D object points
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

objpoints = []
imgpoints = []
valid_images = []

images = sorted(glob.glob(CALIBRATION_DIR + "*.jpg") + 
                glob.glob(CALIBRATION_DIR + "*.png") + 
                glob.glob(CALIBRATION_DIR + "*.jpeg"))

print(f"Found {len(images)} images")
print(f"Looking for {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} inner corners")
print(f"Expected square size: {SQUARE_SIZE*1000:.1f} mm\n")

for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    if img is None:
        print(f"✗ Failed to load: {fname}")
        continue
    
    original_shape = img.shape[:2]
    
    # Resize to target resolution
    img = cv2.resize(img, RESOLUTION)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
    
    if ret:
        # Refine corners
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        
        objpoints.append(objp)
        imgpoints.append(corners2)
        valid_images.append(fname)
        
        # Save debug image with corners drawn
        cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2, ret)
        debug_path = os.path.join(OUTPUT_DIR, f"corners_{idx:03d}.jpg")
        cv2.imwrite(debug_path, img)
        
        print(f"✓ [{len(objpoints):2d}] {os.path.basename(fname)} (original: {original_shape[1]}x{original_shape[0]})")
    else:
        print(f"✗ {os.path.basename(fname)} - corners not found")

print(f"\n{'='*50}")
print(f"VALID IMAGES: {len(objpoints)}/{len(images)}")

if len(objpoints) < 10:
    print("ERROR: Need at least 10 valid images for good calibration!")
    exit(1)

# Calibrate
print("\nCalibrating...")
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, RESOLUTION, None, None
)

# Calculate reprojection error
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error

avg_error = total_error / len(objpoints)

print(f"\n{'='*50}")
print("CALIBRATION RESULTS")
print(f"{'='*50}")
print(f"Resolution: {RESOLUTION[0]}x{RESOLUTION[1]}")
print(f"Valid images used: {len(objpoints)}")
print(f"Reprojection error: {avg_error:.4f} pixels (should be < 0.5)")
print(f"\nFocal length fx: {camera_matrix[0,0]:.2f}")
print(f"Focal length fy: {camera_matrix[1,1]:.2f}")
print(f"Principal point cx: {camera_matrix[0,2]:.2f}")
print(f"Principal point cy: {camera_matrix[1,2]:.2f}")
print(f"\nCamera matrix:\n{camera_matrix}")
print(f"\nDistortion coefficients (k1, k2, p1, p2, k3):")
print(f"{dist_coeffs.ravel()}")

# Save files
np.savetxt("cameraMatrix.txt", camera_matrix, delimiter=',')
np.savetxt("cameraDistortion.txt", dist_coeffs, delimiter=',')
print(f"\n✓ Saved to cameraMatrix.txt and cameraDistortion.txt")
print(f"✓ Debug images saved to {OUTPUT_DIR}/")

# Validation
fx = camera_matrix[0,0]
if 500 < fx < 2000:
    print(f"\n✓ Focal length ({fx:.0f}) looks GOOD for 720p")
elif fx > 3000:
    print(f"\n⚠ WARNING: Focal length ({fx:.0f}) is TOO HIGH!")
    print("  → Images were likely captured at higher resolution than 1280x720")
    print("  → Retake photos at exactly 1280x720")
else:
    print(f"\n⚠ WARNING: Focal length ({fx:.0f}) seems unusual")
