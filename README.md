# Real-Time Object Detection with SSD MobileNet v3

This project demonstrates real-time object detection using the SSD MobileNet v3 model with OpenCV. The model is pre-trained on the COCO dataset and can detect a wide range of objects in live video frames captured from a camera

.![example06_result](https://github.com/ShishirPandy/pythonProject_Objects/assets/87159675/71adb3e3-6836-4b59-84b5-ca33bc0dfa05)


## Requirements

- Python 3.x
- OpenCV (cv2) library
- NumPy library

Install the required packages using pip:

```bash
pip install opencv-python
pip install numpy

##   Clone The Repository

git clone https://github.com/your-username/object-detection-ssd-mobilenet.git
cd object-detection-ssd-mobilenet

Download the pre-trained model files:

Download the 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt' and 'frozen_inference_graph.pb' files and place them in the project directory.
Prepare the 'coco.names' file:

Create a file named 'coco.names' in the project directory and add class labels from the COCO dataset, each label on a new line.
Prepare ground truth annotations (optional):

If you want to measure model performance using precision, recall, and average precision metrics, create an 'annotations.txt' file containing ground truth annotations in the format [class_id, x, y, width, height]. One annotation per line.

