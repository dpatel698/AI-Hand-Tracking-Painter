# AI-Hand-Tracking-Painter

A Python virtual painting program that uses deep learning to recognize hand gestures and enables a user to draw on a webcam video. A fully connected neural network is trained to read landmark data points detected on hands in a video which are then used to map to certain hand gestures.

![](https://github.com/dpatel698/AI-Hand-Tracking-Painter/blob/master/paintergif.gif)

## Features

- Virtual Painter
- Detect Hand Gestures 
- Train Neural Network to Recognize Hand Signs 

## Run Locally

Clone the project

```bash
  git clone https://github.com/dpatel698/AI-Hand-Tracking-Painter
```

Go to the project directory

```bash
  cd AI-Hand-Tracking-Painter
```

Install dependencies

```bash
  pip install opencv-python mediapipe tensorflow
```

Run the painter

```bash
  python itelliPainter.py
```
Right Hand Control:
  1. If the user's right index finger is pointed up the video will be painted or erased based on the mode
     
Left Hand Sign Control:
  1. Stop (five fingers up) - paint red
  1. Peace - paint blue
  3. Fist  - paint green
  4. Thumbs Up - clear canvas of all paint
  5. Spiderman  (index, pinky, thumb up) - toggle eraser mode

Run hand gesture detection

```bash
  python detectHands.py
```

## Feedback

If you have any feedback, please reach out to me at dpatel0698@gmail.com



