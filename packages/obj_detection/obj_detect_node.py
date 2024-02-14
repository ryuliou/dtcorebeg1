import sys
import os
import numpy as np
import cv2
import rospy
import object_detect_lib
from object_detect_lib import Yolo
from std_msgs.msg import Empty
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class ObjectDetectionNode:
    def __init__(self):
        self.__bridge = CvBridge()
        # Publisher to publish predicted image
        self.__image_pub = rospy.Publisher('/'+os.environ['DUCKIEBOT_NAME']+'/prediction_images', Image, queue_size=1)
        # Subscribe to the topic which will supply the image fom the camera
        self.__image_sub = rospy.Subscriber('/'+ os.environ['DUCKIEBOT_NAME'] + "/camera_node/image/raw", Image, self.Imagecallback)

        # Flag to indicate that we have been requested to use the next image
        self.__scan_next = True


        # Create the object_detection_lib class instance
        yolo_c = 'yolo_files\yolov3.cfg'
        yolo_w = 'yolo_files\yolov3.weights'
        self.__odc = object_detection_lib.Yolo(yolo_c, yolo_w) # Yolo object detection model

    # Callback for start command message
    def StartCallback(self, data):
        # Indicate to use the next image for the scan
        self.__scan_next = True


    # Callback for new image received
    def Imagecallback(self, data):
        # if self.__scan_next == True:
        # print("WORKS!")
        rospy.loginfo("Predicting")
        # self.__scan_next = False
        # Convert the ROS image to an OpenCV image
        image = self.__bridge.imgmsg_to_cv2(data, "bgr8")

        # Get bounding boxes of detected objects
        bounding_boxes = self.__odc.detect(image)
        # Draw the bounding boxes on the image
        image = self.__odc.drawBoundingBoxes(image, bounding_boxes)
        # The supplied image will be modified if known objects are detected
        # Predict result, the result will be drawn to `image`
        rospy.loginfo("Finish prediction")
        # publish the image, it may have been modified
        try:
            self.__image_pub.publish(self.__bridge.cv2_to_imgmsg(image, "bgr8"))

        except CvBridgeError as e:
                print(e)

            # Publish names of objects detected
            # result = detection_results()
            # result.names_detected = object_names_detected
            # self.__result_pub.publish(result)

def main(args):
    rospy.init_node('obj_detect_node', anonymous=False)
    # initialize class ObjectDetectionNode
    odc = ObjectDetectionNode()
    rospy.loginfo("YOLO object detection node started")
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)