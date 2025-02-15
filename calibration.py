"""
This script performs camera calibration using a set of images of a chessboard pattern.
It includes functions to calculate the reprojection error, manually select corners if automatic detection fails,
and interpolate the chessboard corners. The main function calibrates the camera and prints the camera intrinsics matrix.

Functions:
- calculate_reprojection_error: Calculates the average reprojection error for the calibration.
- manual_corner_selection: Allows the user to manually select the four corners of the chessboard.
- interpolate_corners: Interpolates the chessboard corners based on the manually selected corners.
- calibrate_camera: Calibrates the camera using the provided images and prints the camera intrinsics matrix.
"""

import numpy as np
import cv2 as cv
import glob
import pickle


def calculate_reprojection_error(
    objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist
):
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist
        )
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        total_error += error
    return total_error / len(objpoints)


def manual_corner_selection(image):
    corners = []

    def click_event(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            corners.append((x, y))
            cv.circle(image, (x, y), 5, (0, 255, 0), -1)

            cv.imshow("Select Corners", image)
            if len(corners) == 4:
                cv.destroyAllWindows()

    cv.imshow("Select Corners", image)
    cv.setMouseCallback("Select Corners", click_event)
    while len(corners) < 4:
        cv.waitKey(1)
    return corners


def interpolate_corners(corners, chessboard_size):
    top_left, top_right, bottom_right, bottom_left = corners
    width, height = chessboard_size

    objp = np.zeros((width * height, 2), np.float32)

    for i in range(height):
        for j in range(width):
            x = (
                top_left[0]
                + j * (top_right[0] - top_left[0]) / (width - 1)
                + i * (bottom_left[0] - top_left[0]) / (height - 1)
            )
            y = (
                top_left[1]
                + j * (top_right[1] - top_left[1]) / (width - 1)
                + i * (bottom_left[1] - top_left[1]) / (height - 1)
            )
            objp[i * width + j] = [x, y]

    return objp


def calibrate_camera(
    images_path, chessboard_size, square_size, frame_size, criteria, error_threshold=1.0
):
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
        img = cv.resize(img, (780,540))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            cv.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv.imshow("img", img)
            cv.waitKey(1000)
        else:
            print(
                "Automatic corner detection failed. Please select the four corners manually."
            )
            print(image)
            corners = manual_corner_selection(img)
            if len(corners) == 4:
                interpolated_corners = interpolate_corners(corners, chessboard_size)
                objpoints.append(objp)
                imgpoints.append(interpolated_corners)
                cv.drawChessboardCorners(
                    img, chessboard_size, interpolated_corners, True
                )
                cv.imshow("img", img)
                cv.waitKey(1000)

    cv.destroyAllWindows()

    # Initial calibration
    ret, camera_matrix, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, frame_size, None, None
    )



    # Print the explicit form of the camera intrinsics matrix K
    print("Camera Intrinsics Matrix K:")
    print(camera_matrix)

    print(f"Initial calibration error: {ret}")

    # Calculate initial re-projection error
    # initial_error = calculate_reprojection_error(
    #     objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist
    # )
    # print(f"Initial re-projection error: {initial_error}")

    # Iteratively remove low-quality images
    # while initial_error > error_threshold:
    #     max_error = 0
    #     max_error_index = -1
    #     for i in range(len(objpoints)):
    #         imgpoints2, _ = cv.projectPoints(
    #             objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist
    #         )
    #         error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    #         if error > max_error:
    #             max_error = error
    #             max_error_index = i
    #
    #     if max_error_index != -1:
    #         print(f"Removing image with highest re-projection error: {max_error}")
    #         del objpoints[max_error_index]
    #         del imgpoints[max_error_index]
    #         rvecs = list(rvecs)
    #         tvecs = list(tvecs)
    #         del rvecs[max_error_index]
    #         del tvecs[max_error_index]
    #
    #         ret, camera_matrix, dist, rvecs, tvecs = cv.calibrateCamera(
    #             objpoints, imgpoints, frame_size, None, None
    #         )
    #         initial_error = calculate_reprojection_error(
    #             objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist
    #         )
    #         print(f"New re-projection error: {initial_error}")

    with open("calibration.pkl", "wb") as f:
        pickle.dump((camera_matrix, dist), f)

    return ret, camera_matrix, dist, rvecs, tvecs



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




if __name__ == "__main__":
    chessboard_size = (9, 6)
    square_size = 22  # in mm
    frame_size = (780,540)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    images_path = "images3/*.png"

    ret, camera_matrix, dist, rvecs, tvecs = calibrate_camera(images_path, chessboard_size, square_size, frame_size, criteria)

    # For drawing the cube on the test image

    print(camera_matrix)

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0: chessboard_size[0], 0: chessboard_size[1]].T.reshape(-1, 2)
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

    img = cv.imread("testimage.jpeg")
    img = cv.resize(img, (780,540))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    ret, rvecs, tvecs = cv.solvePnP(objp, corners2, camera_matrix, dist)
    imgpts, _ = cv.projectPoints(axis_points, rvecs, tvecs, camera_matrix, dist)
    frame = draw_axes(img, corners2, imgpts)

    imgpts, _ = cv.projectPoints(axis, rvecs, tvecs, camera_matrix, dist)
    frame = draw_cube(frame, corners2, imgpts)
    frame = color_top(frame, imgpts, rvecs, tvecs)

    cv.imshow("frame", frame)
    cv.imwrite("testimagewcube3.jpg", frame)
    cv.waitKey(1)



