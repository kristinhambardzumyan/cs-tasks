# Object Detection with YOLOv3

## Requirements
C++ compiler (supporting C++11 or later)

OpenCV library

Pre-trained [yolo3.weights](https://pjreddie.com/darknet/yolo/) file downloaded and placed in the data folder

## Usage
g++ detect-objects-from-webcam.cpp -o output \`pkg-config --cflags --libs opencv4\`

./output
