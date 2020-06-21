# Imports

import numpy as np
from cv2 import resize
from cv2.dnn import readNetFromCaffe, blobFromImage


class FaceDetector:
    """ Class for the face detector model"""

    def __init__(self, prototype, model):
        self.prototype = prototype
        self.model = model
        self.confidence_threshold = 0.6
        self.classifier = readNetFromCaffe(prototype, model)

    def detect(self, image):
        """method to detect faces in input image"""
        classifier = self.classifier
        height, width = image.shape[:2]
        image_blob = blobFromImage(resize(image, (300, 300)), 1.0, (300, 300), (103.93, 116.77, 123.68))
        classifier.setInput(image_blob)
        detections = classifier.forward()
        faces = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the 'confidence' is greater than the minimum confidence
            if confidence > self.confidence_threshold:
                # compute the coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                start_x, start_y, end_x, end_y = box.astype("int")
                # ensuring the bounding boxes fall within the dimensions of the frame
                faces.append(np.array([start_x, start_y, end_x - start_x, end_y - start_y]))

        return faces
