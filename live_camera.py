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


def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)
    # draw pillars in blue
    for i, j in zip(range(4), range(4, 8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 3)
    # draw top layer in red
    img = cv.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
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
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, cameraMatrix, dist)
        frame = draw(frame, corners2, imgpts)

    cv.imshow("frame", frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
