import cv2

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)

thre = 0.45

classess = []
classfile = 'coco.names'

with open(classfile, 'rt') as f:
    classess = [line.rstrip() for line in f]

configpath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightpath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightpath, configpath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classids, conf, bbox = net.detect(img, confThreshold=thre)

    if len(classids) != 0:
        for classid, confidence, box in zip(classids.flatten(), conf.flatten(), bbox):
            # Ensure classid is within the valid range
            if classid - 1 < len(classess):
                label = classess[classid - 1].upper()
            else:
                label = 'UNKNOWN'

            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, label, (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Output", img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

