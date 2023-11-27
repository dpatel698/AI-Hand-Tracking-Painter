# Virtual AI painter application that enables the user to use their finger to draw on a video feed and interact with
# the video using just their hands to clear the canvas, erase paint, and change color with different hand signs
import cv2
import numpy as np
import mediapipe as mp
from model.handSignClassifier import HandSignClassifier
from collectSigns import normalize_landmarks


def paint_canvas(canvas, x, y, brush_color):
    """
    Paints the canvas of a color at the location (x,y).
    :param canvas: image canvas
    :param x: x-coordinate
    :param y: y-coordinate
    :param brush_color: color to paint the canvas
    """
    color_map = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0)}
    color = color_map[brush_color]
    if x > 0 and y > 0:
        cv2.circle(canvas, (x, y), 10, color, cv2.FILLED)


def erase_canvas(canvas, x, y):
    """
    Clears the canvas of a color at the location (x,y).
    :param canvas: image canvas
    :param x: x-coordinate
    :param y: y-coordinate
    """
    if x > 0 and y > 0:
        cv2.circle(canvas, (x, y), 10, (0, 0, 0), cv2.FILLED)


def draw_info(frame, mode):
    if mode == "paint":
        cv2.putText(frame, "Paint Mode", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    if mode == "erase":
        cv2.putText(frame, "Erase Mode", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)


def main():
    """
    AI Virtual Paint Tool
    Right Hand Control:
    1. If the user's right index finger is pointed up the video will be painted or erased based on the mode
    Left Hand Sign Control:
    1. Stop (five fingers up) - paint red
    1. Peace - paint blue
    3. Fist  - paint green
    4. Thumbs Up - clear canvas of all paint
    5. Spiderman  (index, pinky, thumb up) - toggle eraser mode
    """
    # Initialize mediapipe
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils

    # Load the sign recognizer model
    model = HandSignClassifier()

    # Load class names
    f = open('data/handSigns', 'r')
    classNames = f.read().split('\n')
    f.close()

    # Initialize the webcam
    cap = cv2.VideoCapture(1)

    # Set up painter variables
    _, frame = cap.read()
    x, y, c = frame.shape
    canvas = np.zeros((x, y, 3), dtype="uint8")
    brush_color = "red"  # default color for the brush is red
    mode = "paint"

    while True:
        _, frame = cap.read()
        x, y, c = frame.shape

        # Mirror the display
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands in the video
        result = hands.process(framergb)

        # Process the hand detection results
        if result.multi_hand_landmarks:
            landmarks = []
            for hand, handedness in zip(result.multi_hand_landmarks,
                                        result.multi_handedness):
                landmarks = []
                for lm in hand.landmark:
                    landmark_x = int(lm.x * x)
                    landmark_y = int(lm.y * y)
                    landmarks.append([landmark_x, landmark_y])

                if handedness.classification[0].label[0:] == "Left":
                    signClassID = model(normalize_landmarks(landmarks))
                    if signClassID == 0:
                        mode = "paint"
                        brush_color = "red"
                    if signClassID == 1:
                        mode = "paint"
                        brush_color = "blue"
                    if signClassID == 2:
                        mode = "paint"
                        brush_color = "green"
                    if signClassID == 3:
                        mode = "paint"
                        canvas = np.zeros((x, y, 3), dtype="uint8")
                    if signClassID == 4:
                        mode = "erase"
                if handedness.classification[0].label[0:] == "Right":
                    pointer_x, pointer_y = int(hand.landmark[8].x * y), int(hand.landmark[8].y * x)
                    signClassID = model(normalize_landmarks(landmarks))

                    if mode == "paint" and signClassID == 5:
                        paint_canvas(canvas, pointer_x, pointer_y, brush_color)
                    if mode == "erase" and signClassID == 5:
                        erase_canvas(canvas, pointer_x, pointer_y)

                # Draw hand landmarks on frames
                mpDraw.draw_landmarks(frame, hand,
                                      mpHands.HAND_CONNECTIONS)

                # Put text info about the state of the application onto the frame
                draw_info(frame, mode)

        # Combine the canvas and the video frame
        frame_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, frame_inverse = cv2.threshold(frame_gray, 50, 255, cv2.THRESH_BINARY_INV)
        frame_inverse = cv2.cvtColor(frame_inverse, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, frame_inverse)
        frame = cv2.bitwise_or(frame, canvas)

        # Put text info on the video
        draw_info(frame, mode)

        # Show the final output
        cv2.imshow("Output", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
