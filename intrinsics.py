import cv2
import numpy as np
import os

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
objp[:, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Number of frames to process per video
NUM_FRAMES_TO_USE = 50

# Process each camera
camera_dirs = ["data/cam1/", "data/cam2/", "data/cam3/", "data/cam4/"]

for cam_dir in camera_dirs:
    video_path = os.path.join(cam_dir, "intrinsics.avi")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES_TO_USE, dtype=int)

    objpoints = []  # 3D world points
    imgpoints = []  # 2D image points

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria=(
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                ),
            )
            imgpoints.append(corners2)

    cap.release()

    if len(objpoints) > 0:
        ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )

        # Save intrinsics to XML file
        intrinsics_file = os.path.join(cam_dir, "intrinsics_cam.xml")
        fs = cv2.FileStorage(intrinsics_file, cv2.FILE_STORAGE_WRITE)
        fs.write("CameraMatrix", camera_matrix)
        fs.write("DistortionCoeffs", dist_coeffs)
        fs.release()

        print(f"Saved intrinsic parameters for {cam_dir} to {intrinsics_file}")
    else:
        print(f"No valid chessboard detections for {cam_dir}")
