"""
This script captures live video from the webcam and performs real-time processing to draw 3D axes and a cube on a detected chessboard pattern.
It uses previously saved camera calibration data to project 3D points onto the 2D image plane.

Steps:
- Load camera calibration data from a file.
- Define the chessboard pattern size and criteria for corner refinement.
- Prepare object points for the chessboard pattern.
- Capture live video from the webcam.
- Detect the chessboard pattern in each frame.
- Draw 3D axes and a cube on the detected chessboard pattern.
- Color the top side of the cube based on its distance and orientation relative to the camera.
"""


import cv2 as cv
import numpy as np
import pickle

# Load previously saved camera calibration data
with open("calibration.pkl", "rb") as f:
    cameraMatrix, dist = pickle.load(f)

# Define the chessboard size
chessboardSize = (9, 6)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : chessboardSize[0], 0 : chessboardSize[1]].T.reshape(-1, 2)
size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm

# Define the cube points in 3D
axis = np.float32(
    [
        [0, 0, 0],
        [0, size_of_chessboard_squares_mm, 0],
        [size_of_chessboard_squares_mm, size_of_chessboard_squares_mm, 0],
        [size_of_chessboard_squares_mm, 0, 0],
        [0, 0, -size_of_chessboard_squares_mm],
        [0, size_of_chessboard_squares_mm, -size_of_chessboard_squares_mm],
        [
            size_of_chessboard_squares_mm,
            size_of_chessboard_squares_mm,
            -size_of_chessboard_squares_mm,
        ],
        [size_of_chessboard_squares_mm, 0, -size_of_chessboard_squares_mm],
    ]
)

# Define the 3D points for the axes
axis_points = np.float32([[0, 0, 0], [50, 0, 0], [0, 50, 0], [0, 0, -50]])


def draw_axes(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    corner = tuple(map(int, corners[0].ravel()))
    img = cv.line(img, corner, tuple(imgpts[1]), (0, 0, 255), 5)  # X-axis in red
    img = cv.line(img, corner, tuple(imgpts[2]), (0, 255, 0), 5)  # Y-axis in green
    img = cv.line(img, corner, tuple(imgpts[3]), (255, 0, 0), 5)  # Z-axis in blue
    return img


def draw_cube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)
    # draw pillars in blue
    for i, j in zip(range(4), range(4, 8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 3)
    # draw top layer in red
    img = cv.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
    return img


def color_top(img, imgpts, rvecs, tvecs):
    # Calculate distance to the camera
    distance = np.linalg.norm(tvecs)
    v_value = max(0, min(255, 255 * (1 - distance / 4000)))

    # Calculate orientation
    rotation_matrix, _ = cv.Rodrigues(rvecs)
    z_axis = rotation_matrix[:, 2]
    angle = np.arccos(np.dot(z_axis, np.array([0, 0, 1])))
    s_value = max(0, min(255, 255 * (1 - angle / (np.pi / 4))))

    # Calculate hue based on position
    hue = int((tvecs[0][0] + 2000) % 180)  # Extract the single element from tvecs

    # Convert HSV to BGR
    color = cv.cvtColor(np.uint8([[[hue, s_value, v_value]]]), cv.COLOR_HSV2BGR)[0][0]

    # Ensure imgpts are in the correct format
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # Draw the top side of the cube
    img = cv.fillConvexPoly(img, imgpts[4:], color.tolist())
    return img


cap = cv.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    if ret:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, cameraMatrix, dist)
        imgpts, _ = cv.projectPoints(axis_points, rvecs, tvecs, cameraMatrix, dist)
        frame = draw_axes(frame, corners2, imgpts)

        imgpts, _ = cv.projectPoints(axis, rvecs, tvecs, cameraMatrix, dist)
        frame = draw_cube(frame, corners2, imgpts)
        frame = color_top(frame, imgpts, rvecs, tvecs)

    cv.imshow("frame", frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
