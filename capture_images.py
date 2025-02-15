"""
This script captures images from a video source and saves them to a specified directory.
"""

import cv2
import os


def create_images_directory(directory="images"):
    if not os.path.exists(directory):
        os.makedirs(directory)


def capture_images(video_source=0, directory="images"):
    create_images_directory(directory)
    cap = cv2.VideoCapture(video_source)
    num = 16

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        k = cv2.waitKey(5)
        if k == 27:
            break
        elif k == ord("s"):  # wait for 's' key to save and exit
            cv2.imwrite(f"{directory}/img{num}.png", img)
            print("image saved!")
            num += 1

        cv2.imshow("Img", img)

    # Release and destroy all windows before termination
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_images()
