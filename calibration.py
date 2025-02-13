import numpy as np
import cv2 as cv
import glob
import pickle


def calibrate_camera(images_path, chessboard_size, square_size, frame_size, criteria):
    # Prepare object points
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(
        -1, 2
    )
    objp *= square_size

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    images = glob.glob(images_path)

    for image in images:
        img = cv.imread(image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            cv.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv.imshow("img", img)
            cv.waitKey(1000)

    cv.destroyAllWindows()

    ret, camera_matrix, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, frame_size, None, None
    )

    with open("calibration.pkl", "wb") as f:
        pickle.dump((camera_matrix, dist), f)

    return ret, camera_matrix, dist, rvecs, tvecs


if __name__ == "__main__":
    chessboard_size = (9, 6)
    square_size = 22  # in mm
    frame_size = (1280, 720)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    images_path = "images/*.png"

    calibrate_camera(images_path, chessboard_size, square_size, frame_size, criteria)
