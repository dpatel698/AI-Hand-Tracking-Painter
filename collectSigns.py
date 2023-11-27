# This module collects hand landmark data for hand signs in a video feed and appends them to a csv file that
# can be used to train a model to detect those signs
import cv2
import mediapipe as mp
import sys
import csv
from os.path import exists

def append_to_csv(class_label, normalized_landmarks):
    """
    Save a new data point to a csv file.

    :param class_label: the number for the class
    :param normalized_landmarks: the hand landmarks normalized to the [0, 1] range
    """
    csv_path = 'data/signData.csv'
    if not exists(csv_path):
        open(csv_path, "x")
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([class_label, *normalized_landmarks])

def normalize_landmarks(landmarks):
    """
    Take each hand landmark, convert to relative coordinates, and normalize to the [0,1] range using linear scaling
    :param landmarks: the hand landmarks list
    :return: normalized landmarks
    """
    # Convert to relative coordinates
    rel_x, rel_y = landmarks[0][0], landmarks[0][1]
    relative_landmarks = [[x - rel_x, y - rel_y] for x, y in landmarks]

    # Flatten the list
    relative_landmarks = [num for sublist in landmarks for num in sublist]

    # Normalize the coordinates
    max_value = max(map(abs, relative_landmarks))
    min_value = min(map(abs, relative_landmarks))
    relative_landmarks = list(map(lambda num: (num - min_value) / (max_value - min_value), relative_landmarks))

    return relative_landmarks


def main():
    # Read system args to get the class number
    if len(sys.argv) < 2:
        print("-classLabel not given as argument")
        return
    class_label = int(sys.argv[1])

    # Initialize mediapipe
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils

    # Initialize the webcam for Hand Sign Recognition
    cap = cv2.VideoCapture(1)
    while True:
        _, frame = cap.read()
        x, y, c = frame.shape

        # Mirror the display
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if cv2.waitKey(1) == ord('q'):
            break
        if cv2.waitKey(50) == ord('c'):
            # If hand found then save the hand landmark data to a csv along with the chosen label for the class
            # Get hand landmark prediction
            result = hands.process(framergb)

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

                    # Save normalized landmark data to csv
                    normalized_landmarks = normalize_landmarks(landmarks)
                    append_to_csv(class_label, normalized_landmarks)

                    # Drawing hand landmarks on frames
                    mpDraw.draw_landmarks(frame, hand,
                                          mpHands.HAND_CONNECTIONS)
                    cv2.imshow("Video", frame)
                    cv2.waitKey(500)
        if cv2.waitKey(1) == ord('1'):
            # Change the class label to the given class label number
            print("Enter a number for the class label: ")
            class_label = int(input())
        else:
            cv2.imshow("Video", frame)

    # release the webcam and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


