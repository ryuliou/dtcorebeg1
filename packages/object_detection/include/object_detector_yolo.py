
import numpy as np
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from duckietown_msgs.msg import Segment, SegmentList, AntiInstagramThresholds
from line_detector import LineDetector, ColorRange, plotSegments, plotMaps
from image_processing.anti_instagram import AntiInstagram

from duckietown.dtros import DTROS, NodeType, TopicType


class ObjectDetectorYolo(DTROS):

    def __init__(self, node_name):
        #  Initialize the DTROS parent class
        super(ObjectDetectorYolo, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        









