       
class BoundingBoxes:
    """
    This is a data class that can be used to store the results of YOLO detection procedure performed by :py:class:`LineDetector`.
    
    """

    def __init__(self, bounding_boxes):
        self.bounding_boxes = bounding_boxes #: An ``Nx4`` array with every row representing a bounding box [x_min, y_min, box_width, box_height]
