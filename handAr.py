# This program takes in camera calibration parameters from csv files, detects hands, and projects AR objects onto
# hands
import cv2
import numpy as np
import mediapipe as mp


def draw_cube(img, imgpts):
    """
    Citation: https://docs.opencv.org/4.x/d7/d53/tutorial_py_pose.html
    """
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)
    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
    return img


def main():
    # Initialize the camera and distortion matrices from saved csv files
    try:
        camera_matrix = np.loadtxt("cameraMatrix.csv", delimiter=",")
        dist_matrix = np.loadtxt("distortionMatrix.csv", delimiter=",")
    except FileNotFoundError:
        print("Camera calibration files missing")

    # Set up hand recognition
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((5 * 4, 3), np.float32)
    objp[:, :2] = np.mgrid[0:5, 0:4].T.reshape(-1, 2)

    # Run loop to recognize hand and project an object onto it
    cap = cv2.VideoCapture(1)
    while True:
        # Read each frame from the webcam
        _, frame = cap.read()
        x, y, c = frame.shape

        # Mirror the display
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get hand landmark prediction
        result = hands.process(framergb)
        className = ''

        # Process the hand results
        if result.multi_hand_landmarks:
            landmarks = []
            for hand, handedness in zip(result.multi_hand_landmarks,
                                        result.multi_handedness):
                landmarks = []
                for lm in hand.landmark:
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)
                    landmarks.append([lmx, lmy])

                # Draw hand landmarks on frames
                mpDraw.draw_landmarks(frame, hand,
                                      mpHands.HAND_CONNECTIONS)

                # Project hand and object points onto image
                axis = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
                                   [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])
                landmarks_arr = np.float32(landmarks[1:])
                ret, rvecs, tvecs = cv2.solvePnP(objp, landmarks_arr, camera_matrix, dist_matrix)
                imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, camera_matrix, dist_matrix)

                # Only draw if the cube is rendered well (axis isn't stretched too large during projection)
                if imgpts[1][0][1] - imgpts[4][0][1] < y * .1:
                    draw_cube(frame, imgpts)

        # Show the final output
        cv2.imshow("Output", frame)
        if cv2.waitKey(200) == ord('q'):
            break


if __name__ == "__main__":
    main()
