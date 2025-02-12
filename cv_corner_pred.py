import cv2
import numpy as np
import os

# Set the folder where images are stored
image_folder = "images"
output_folder = "output_corners"
os.makedirs(output_folder, exist_ok=True)

# Chessboard settings
chessboard_size = (7, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Log file for failed detections
log_file = os.path.join(output_folder, "detection_log.txt")
failed_images = []

# Process each image in the folder
for filename in os.listdir(image_folder):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # # show gray image
        # cv2.imshow("gray", gray)
        # cv2.waitKey(0)

        # Try to detect chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            # Refine detected corners
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )

            # Draw and save detected corners
            img_with_corners = cv2.drawChessboardCorners(
                img, chessboard_size, corners_refined, ret
            )
            cv2.imwrite(
                os.path.join(output_folder, f"detected_{filename}"), img_with_corners
            )
            print(f"Detected corners in {filename}")
        else:
            failed_images.append(filename)
            print(f"Failed to detect corners in {filename}")

# Save failed detections log
with open(log_file, "w") as f:
    f.write("Failed Images (Manual Annotation Needed):\n")
    f.writelines("\n".join(failed_images))

print("\nProcessing complete. Check the output folder for results.")
