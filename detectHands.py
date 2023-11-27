# Citation: https://techvidvan.com/tutorials/hand-gesture-recognition-tensorflow-opencv/
# This file contains code for hand detection and sign recognition used for the base of this project
import cv2
import numpy as np
import mediapipe as mp
from model.handSignClassifier import HandSignClassifier
from collectSigns import normalize_landmarks

def get_bounding_box(frame, landmarks):
    """
    Calculates the coordinates for the bounding box
    """
    image_width, image_height = frame.shape[1], frame.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def display_bounding_box(image, bbox):
    """
    Draws a bounding box around the hand
    """
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                  (0, 0, 0), 1)

    return image

def show_hand_info(frame, bbox, handedness, hand_sign):
    """
    Draws the bounding box, hand label and pose for a hand in the image
    """
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[1] - 22),
                  (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign != "":
        info_text = info_text + ':' + hand_sign
    cv2.putText(frame, info_text, (bbox[0] + 5, bbox[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def main():
    # Initialize mediapipe
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.8)
    mpDraw = mp.solutions.drawing_utils

    # Load the sign recognizer model
    model = HandSignClassifier()

    # Load class names
    f = open('data/handSigns', 'r')
    signNames = f.read().split('\n')

    # Initialize the webcam for Hand Sign Recognition
    cap = cv2.VideoCapture(1)
    while True:
        _, frame = cap.read()
        x, y, c = frame.shape

        # Mirror the display
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get hand landmark prediction
        result = hands.process(frame_rgb)

        # Process the hand results
        if result.multi_hand_landmarks:
            landmarks = []
            for hand, handedness in zip(result.multi_hand_landmarks,
                                                  result.multi_handedness):
                landmarks = []
                for lm in hand.landmark:
                    landmark_x = int(lm.x * x)
                    landmark_y = int(lm.y * y)
                    landmarks.append([landmark_x, landmark_y])

                # Draw hand landmarks on frame
                mpDraw.draw_landmarks(frame, hand,
                                      mpHands.HAND_CONNECTIONS)

                # Predict sign and display text
                signClassID = model(normalize_landmarks(landmarks))
                signName = signNames[signClassID]

                # Draw Bounding Box and prediction label for the hand
                bbox = get_bounding_box(frame, hand)
                display_bounding_box(frame, bbox)
                show_hand_info(frame, bbox, handedness, signName)

        # Show the final output
        cv2.imshow("Output", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
