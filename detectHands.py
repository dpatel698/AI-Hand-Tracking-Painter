import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model


def main():
    # Initialize mediapipe
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils

    # Load the gesture recognizer model
    model = load_model('mp_hand_gesture')

    # Load class names
    f = open('gesture.names', 'r')
    classNames = f.read().split('\n')
    f.close()
    print(classNames)

    # Initialize the webcam for Hand Gesture Recognition Python project
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
            for idx, hand in enumerate(result.multi_hand_landmarks):
                for lm in hand.landmark:
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)
                    print((lmx, lmy))
                    landmarks.append([lmx, lmy])
                # Drawing landmarks on frames
                mpDraw.draw_landmarks(frame, hand,
                                      mpHands.HAND_CONNECTIONS)

        # Show the final output
        cv2.imshow("Output", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    # release the webcam and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
