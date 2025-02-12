import cv2
import numpy as np

global points
points = []


def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 20, (0, 0, 255), -1)
        cv2.imshow("Manual Corner Selection", img)


# Load chessboard image
img = cv2.imread("calib-checkerboard.png")  # Change filename as needed
cv2.imshow("Manual Corner Selection", img)
cv2.setMouseCallback("Manual Corner Selection", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(points) == 4:
    print("Selected Points:", points)
    # Perform linear interpolation here to get inner corners
else:
    print("Error: Select exactly 4 points!")

if len(points) == 4:
    print("Selected Points:", points)

    # Perform linear interpolation to estimate inner corners
    def interpolate_corners(p1, p2, num_points):
        return [
            (
                int(p1[0] + (p2[0] - p1[0]) * i / (num_points - 1)),
                int(p1[1] + (p2[1] - p1[1]) * i / (num_points - 1)),
            )
            for i in range(num_points)
        ]

    grid_size = (7, 7)  # Assuming 7x7 inner corners
    interpolated_corners = []

    for i in range(grid_size[1]):
        row_points = interpolate_corners(points[0], points[1], grid_size[0])
        col_points = interpolate_corners(points[2], points[3], grid_size[0])
        interpolated_corners.append(
            interpolate_corners(row_points[i], col_points[i], grid_size[0])
        )

    print("Interpolated Corners:", interpolated_corners)
else:
    print("Error: Select exactly 4 points!")


# TODO: the code is not stopping at 4 time
