## Resources used

* https://github.com/niconielsen32/CameraCalibration

## TODO

print coeffs before saving to pickle + 

"Provide the explicit form of camera intrinsics matrix K. This can also be done by an OpenCV-function, figure out which one. Also state the resolution of your (training and test) images."

* merge calibration.py with label.py (when corners not found show image and do manual points and save) else if ret==false
* add code which runs calibration 3 times

* draw axes
* draw cube at the start of coordinates
* color top side of coordinates

* write comments to each script it's purpose, what it does
* prepare assignment report


### CHOICE Tasks

* We already have 1

- CHOICE 1: real-time performance with webcam in online phase: 10
- CHOICE 2: iterative detection and rejection of low quality input images in offline phase: 10. Check for a function output that could provide an indication of the quality of the calibration.
- CHOICE: 3 improving the localization of the corner points in your manual interface: 10. Make the estimation of (a) the four corner points or (b) the interpolated points more accurate.
- CHOICE 4: implement a function that can provide a confidence for how well each variable has been estimated, perhaps by considering subsets of the input: 10
- CHOICE 5: implement a way to enhance the input to reduce the number of input images that are not correctly processed by findChessboardCorners, for example by enhancing edges or getting rid of light reflections: 10
- CHOICE 6: produce a 3D plot with the locations of the camera relative to the chessboard when each of the training images was taken: 10


## REPORT

Report should have 4 points
1. I print "explicit form of camera intrinsics matrix K" 
2. TODO
3. TODO

4. Chosen tasks:
    1) in live_camera.py we test online phase using video frames (cv.VideoCapture)
    2) we calculate reprojection error for each image in calibration.py and set up threshold. If we don't meet required threshold, we eliminate images with highest reprojection error until we do. To setup threshold value it would be good to run calibration multiple times and look into its distribution, ideally it should eliminate mostly outliers. 