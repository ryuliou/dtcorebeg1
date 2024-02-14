import numpy as np
import cv2
from yolo_detections import BoundingBoxes

class Yolo:
     

    def __init__(self, yolo_c, yolo_w):
        """
        Initializes the YOLO object detection model.

        Args:
            yolo_c (:obj:`str`): path to the YOLO configuration file
            yolo_w (:obj:`str`): path to the YOLO weights file
        
        """
        self.yolo_c = yolo_c
        self.yolo_w = yolo_w
        self.network = cv2.dnn.readNetFromDarknet(yolo_c, yolo_w)
        self.yolo_layers = ['yolo_82', 'yolo_94', 'yolo_106']

    def detect(self, img_input):
        """
        Detects objects in an image using YOLO.

        Args:
            img_input (:obj:`numpy array`): input image

        Returns:
            :obj:`list` of :obj:`list` of :obj:`int`: a list of lists containing the bounding boxes of the detected objects
        """

        # Yolo block
        image = cv2.imread(img_input)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert image to blob (4D numpy array [images,channels,width,height])
        input_blob = cv2.dnn.blobFromImage(image_rgb, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.network.setInput(input_blob)
        # Pass through network and start inference timer
        output = network.forward(self.yolo_layers)
        # Define Variables for drawing on image
        bounding_boxes = []
        confidences = []
        classes = []
        probability_minimum = 0.87
        threshold = 0.5
        h, w = image.shape[:2]
        for result in output:
            for detection in result:
                scores = detection[5:]
                class_current = np.argmax(scores)
                # Class current is 13 for a stop sign
                if class_current == 13:
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
        # Store Yolo detections
        bounding_boxes_output = []
        if len(results) > 0:
            for i in results.flatten():
                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
                bounding_boxes_output.append([x_min, y_min, box_width, box_height])
        # Return bounding boxes
        return bounding_boxes_output
    



    def detectBoundingBox(self, img_input):
        """
        Detects the bounding boxes of the objects in the currently set image.

        Returns:
            :obj:`list` of :obj:`list` of :obj:`int`: a list of lists containing the bounding boxes of the detected objects
        """
        bounding_boxes = self.detect(img_input)
        return bounding_boxes
    

    def drawBoundingBoxes(self, img_input, bounding_boxes):
        """
        Draws the bounding boxes of the detected objects on the currently set image.

        Args:
            img_input (:obj:`numpy array`): input image
            bounding_boxes (:obj:`BoundingBoxes`): the bounding boxes of the detected objects

        Returns:
            :obj:`numpy array`: the image with the bounding boxes drawn on top of it.
        """
        image = np.copy(img_input)
        for i,box in bounding_boxes:
            x_min, y_min = box[i][0], box[i][1]
            box_width, box_height = box[i][2], box[i][3]
            colour_box = (255,0,0)
            cv2.rectangle(image, (x_min, y_min), (x_min + box_width, y_min + box_height), colour_box, 5)
        return image







