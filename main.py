# we will import computer vision library for performing the object detetction/Image segmentation or Image Classification
import cv2
# here to use the downloaded/saved video we can use this code in which we are providing the location of video 
 ## video_path = 'your_video_file_path.mp4'
##  cap = cv2.VideoCapture(video_path)

# In this I have capture the video using the live camera so I have provide the dimesnsion of my webcam area which I will be requiring  in the project 

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set camera width to 640 pixels
cap.set(4, 480)  # Set camera height to 480 pixels

## this threeshold istype of accuracy which means if any image or box matches with any weight and names(coco.names) more then this
much percentage then it means the prediction is correct.
    
thre = 0.5  # Threshold value for confidence to filter detections

## now here we will create a list in which we will be storing the class labels as they are stored in coco.names
file without any indexing so in list they will be stored on the basis of index and based on that prediction will be done 
classess = []  # List to store class labels
classfile = 'coco.names'  # File containing class labels for COCO dataset

# Load class labels from the coco.names file into the classess list

with open(classfile, 'rt') as f:
    classess = [line.rstrip() for line in f]

## In below lines, the script sets the paths to the model configuration file (configpath) and model weights file (weightpath). 
#The pre-trained SSD MobileNet model for object detection is then loaded using cv2.dnn_DetectionModel provided by OpenCV's Deep Neural Network (DNN) module. 
#This function takes the weights and configuration files as arguments and initializes the model for further use.

configpath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'  # Model configuration file
weightpath = 'frozen_inference_graph.pb'  # Model weights file

# Load the pre-trained SSD MobileNet model for object detection so that we can use this in our model for detecting the Objects
net = cv2.dnn_DetectionModel(weightpath, configpath)

#the four functions which are used below are for setting the input data to the SSD MobileNet model perform the following operations:
#setInputSize(320, 320): Resizes the input images to a fixed size of 320x320 pixels.
#setInputScale(1.0 / 127.5): Normalizes the pixel values from the range 0-255 to the range -1 to 1.
#setInputMean((127.5, 127.5, 127.5)): Subtracts the mean value of (127.5, 127.5, 127.5) from the input data to center the pixel values around zero.
#setInputSwapRB(True): Swaps the Red and Blue channels of the input data from BGR to RGB format as expected by the model.
#These preprocessing steps ensure that the input data is properly formatted and prepared for accurate and efficient object detection using the SSD MobileNet model.

net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Start the main loop for capturing and processing frames
# This loop reads frames from the video capture (cap) one by one. cap.read() returns a tuple with two values:
# a boolean (success) indicating whether the frame was read successfully and the frame itself (img).
while True:
    # Read a frame from the camera
    success, img = cap.read()

    # Use the model to detect objects in the frame
    classids, conf, bbox = net.detect(img, confThreshold=thre)

    # Check if any objects are detected
    if len(classids) != 0:
        
         # Loop through all detected objects
        #In this line, the model is used to detect objects in the current frame (img). net.detect() returns three arrays:
        #classids (IDs of the detected classes), conf (confidence scores of the detections), and bbox (bounding boxes of the detections).
        
        for classid, confidence, box in zip(classids.flatten(), conf.flatten(), bbox):
            
            # Ensure classid is within the valid range of the classess list
            #This block checks if any objects are detected in the current frame. If there are detections, 
            #it loops through each detection using the zip function to iterate over classids, conf, and bbox simultaneously.
            
            if classid - 1 < len(classess):
                # Get the corresponding class label from classess list
                label = classess[classid - 1].upper()
            else:
                # If classid is out of range, assign 'UNKNOWN' label
                label = 'UNKNOWN'
                
            # Draw a rectangle around the detected object
            #This line draws a rectangle around the detected object on the img frame using the bounding box coordinates provided in box.

            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)

            # Put the class label and confidence on the image
            # These lines add text to the img frame indicating the class label and confidence of each detected object. The label is placed at (box[0] + 10, box[1] + 30) position, 
            # and the confidence is placed at (box[0] + 200, box[1] + 30) position. The font size is 1, and the color is set to green (0, 255, 0) with a thickness of 2.

            cv2.putText(img, label, (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Display the output image with detections
    cv2.imshow("Output", img)

    # Wait for a key press and check if 'q' is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows ,if we are using the live camera feature only then it is executable else not .
cap.release()
cv2.destroyAllWindows()
