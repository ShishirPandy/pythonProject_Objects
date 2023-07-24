import cv2

# Open the camera capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set camera width to 640 pixels
cap.set(4, 480)  # Set camera height to 480 pixels

thre = 0.5  # Threshold value for confidence to filter detections

classess = []  # List to store class labels
classfile = 'coco.names'  # File containing class labels for COCO dataset

# Load class labels from the coco.names file into the classess list
with open(classfile, 'rt') as f:
    classess = [line.rstrip() for line in f]

configpath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'  # Model configuration file
weightpath = 'frozen_inference_graph.pb'  # Model weights file

# Load the pre-trained SSD MobileNet model for object detection
net = cv2.dnn_DetectionModel(weightpath, configpath)

# Set the input size and scale for the model
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Start the main loop for capturing and processing frames
while True:
    # Read a frame from the camera
    success, img = cap.read()

    # Use the model to detect objects in the frame
    classids, conf, bbox = net.detect(img, confThreshold=thre)

    # Check if any objects are detected
    if len(classids) != 0:
        # Loop through all detected objects
        for classid, confidence, box in zip(classids.flatten(), conf.flatten(), bbox):
            # Ensure classid is within the valid range of the classess list
            if classid - 1 < len(classess):
                # Get the corresponding class label from classess list
                label = classess[classid - 1].upper()
            else:
                # If classid is out of range, assign 'UNKNOWN' label
                label = 'UNKNOWN'

            # Draw a rectangle around the detected object
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)

            # Put the class label and confidence on the image
            cv2.putText(img, label, (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Display the output image with detections
    cv2.imshow("Output", img)

    # Wait for a key press and check if 'q' is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
