import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET

# Load checkerboard parameters from XML
fs = cv2.FileStorage("data/checkerboard.xml", cv2.FILE_STORAGE_READ)
CHECKERBOARD = (
    int(fs.getNode("CheckerBoardWidth").real()),
    int(fs.getNode("CheckerBoardHeight").real()),
)
SQUARE_SIZE = (
    fs.getNode("CheckerBoardSquareSize").real() / 1000.0
)  # Convert mm to meters
fs.release()

# Prepare object points (3D points in real-world coordinates)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Axis points for visualization (x=red, y=green, z=blue)
axis_length = SQUARE_SIZE * (CHECKERBOARD[0] - 1)  # Match the length to the full axis
axis = np.float32([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, -axis_length]])

# Process each camera
camera_dirs = ['data/cam1/', 'data/cam2/', 'data/cam3/', 'data/cam4/']

for cam_dir in camera_dirs:
    video_path = os.path.join(cam_dir, 'checkerboard.avi')
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        continue
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)  # Pick a middle frame for calibration
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Could not read a frame from {video_path}")
        continue
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    if not ret:
        print(f"Error: Could not find chessboard corners in {video_path}")
        continue
    
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    
    # Load intrinsic parameters
    intrinsics_file = os.path.join(cam_dir, 'intrinsics_cam.xml')
    fs = cv2.FileStorage(intrinsics_file, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("CameraMatrix").mat()
    dist_coeffs = fs.getNode("DistortionCoeffs").mat()
    fs.release()
    
    # Solve for extrinsic parameters
    ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs)
    
    if not ret:
        print(f"Error: Could not compute extrinsics for {cam_dir}")
        continue
    
    # Project axis points to visualize x, y, z axes
    imgpts, _ = cv2.projectPoints(axis, rvecs, tvecs, camera_matrix, dist_coeffs)
    origin = tuple(imgpts[0].ravel().astype(int))
    x_axis = tuple(imgpts[1].ravel().astype(int))
    y_axis = tuple(imgpts[2].ravel().astype(int))
    z_axis = tuple(imgpts[3].ravel().astype(int))
    
    cv2.line(frame, origin, x_axis, (0, 0, 255), 3)  # X-axis (Red)
    cv2.line(frame, origin, y_axis, (0, 255, 0), 3)  # Y-axis (Green)
    cv2.line(frame, origin, z_axis, (255, 0, 0), 3)  # Z-axis (Blue)
    cv2.circle(frame, origin, 5, (0, 255, 255), -1)  # Yellow dot at (0,0,0)

    
    # Show visualization
    cv2.imshow(f"Extrinsic Calibration - {cam_dir}", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save extrinsics to XML file
    extrinsics_file = os.path.join(cam_dir, 'config.xml')
    fs = cv2.FileStorage(extrinsics_file, cv2.FILE_STORAGE_WRITE)
    fs.write("RotationVector", rvecs)
    fs.write("TranslationVector", tvecs)
    fs.release()
    
    print(f"Saved extrinsic parameters for {cam_dir} to {extrinsics_file}")


# TODO: cam3 green Y-axis line length is too long