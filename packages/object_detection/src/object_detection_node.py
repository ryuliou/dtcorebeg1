
import numpy as np
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from duckietown_msgs.msg import Segment, SegmentList, AntiInstagramThresholds
from line_detector import LineDetector, ColorRange, plotSegments, plotMaps
from image_processing.anti_instagram import AntiInstagram

from duckietown.dtros import DTROS, NodeType, TopicType


yolo_w = ""
yolo_c = ""

# Load YOLO Network
network = cv2.dnn.readNetFromDarknet(yolo_c,
                                    yolo_w)
yolo_layers = ['yolo_82', 'yolo_94', 'yolo_106']

img_names = []

if(len(img_names) > 0):
        for i in range(0,len(img_names)):
            # read image
            image = cv2.imread(img_names[i])

            # Convert image to blob (4D numpy array [images,channels,width,height])
            input_blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            network.setInput(input_blob)
            # Pass through network and start inference timer
            output = network.forward(yolo_layers)

            # Define Variables for drawing on image
            bounding_boxes = []
            confidences = []
            classes = []
            probability_minimum = 0.87
            threshold = 0.5
            h, w = image.shape[:2]

            # Get bounding boxes, confidences, and classes
            for result in output:
                for detection in result:
                    scores = detection[5:]
                    class_current = np.argmax(scores)
                    confidence_current = scores[class_current]
                    if confidence_current > probability_minimum:
                        box_current = detection[0:4] * np.array([w, h, w, h])
                        x_center, y_center, box_width, box_height = box_current.astype('int')
                        x_min = int(x_center - (box_width / 2))
                        y_min = int(y_center - (box_height / 2))
                        bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                        confidences.append(float(confidence_current))
                        classes.append(class_current)


            # Draw bounding boxes and information on image
            results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)
            coco_labels = 80
            # Get classes using the set function
            unique_classes = list(set(classes))
            
            # Output images with bounding boxes 
            np.random.seed(42)
            colours = np.random.randint(0, 255, size=(coco_labels, 3), dtype='uint8')
            if len(results) > 0:
                for i in results.flatten():
                    x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                    box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
                    colour_box = [int(j) for j in colours[classes[i]]]
                    cv2.rectangle(image, (x_min, y_min), (x_min + box_width, y_min + box_height),
                                colour_box, 5)
                    text_box = 'conf: {:.4f}'.format(confidences[i])
           
            cv2.imwrite(out_image_path, image )
