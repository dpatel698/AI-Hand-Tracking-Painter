# Calibrate the camera and save calibration parameters to a csv file
import cv2
import numpy as np


def main():
    """
    Main function for camera calibration. Here are the key presses to interact with the video:
    1. Press 'c' to find chessboard and save parameters found in the image
    2. Press 's' to save calibration matrices if >= 5 chessboard images found
    3. Press 'q' to quit the program
    """
    calibration_count = 0

    # Initialize the webcam for Hand Gesture Recognition Python project
    cap = cv2.VideoCapture(1)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    while True:
        # Read each frame from the webcam
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        key = cv2.waitKey(100)
        if key == ord('q'):
            break
        if key == ord('c'):
            # Capture calibration image data if chessboard found in frame
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)

            # If found, add object points, image points (after refining them)
            if ret:
                calibration_count += 1
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners)
                # Draw and display the corners
                cv2.drawChessboardCorners(frame, (7, 6), corners2, ret)
                cv2.imshow('Video', frame)
                cv2.waitKey(500)
        if key == ord('s') and calibration_count >= 5:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            # Print calibration error
            mean_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                mean_error += error
            print("total error: {}".format(mean_error / len(objpoints)))

            # Save the camera and distortion matrix data to csv files
            np.savetxt("cameraMatrix.csv", mtx, delimiter=",")
            np.savetxt("distortionMatrix.csv", dist, delimiter=",")
        else:
            cv2.imshow('Video', frame)

    # release the webcam and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

