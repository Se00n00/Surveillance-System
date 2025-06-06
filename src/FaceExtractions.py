import cv2
import numpy as np

class FaceExtractions:
    """
    This class extracts face patches from images using a face detector.
    """

    def __init__(self, face_detector):
        self.face_detector = face_detector  # e.g., a YOLO-based detector

    def extract_patches(self, image:np.ndarray)-> list:
        """
        Extracts face patches from the given image using the face detector.

        Args:
            image (numpy.ndarray): The input image from which to extract faces.

        Returns:
            list: A list of extracted face patches (numpy arrays).
        """

        bounding_boxes = self.face_detector.detect_faces(image)
        patches = self.get_patches(image, bounding_boxes)
        return patches

    def get_patches(self, image, bounding_boxes):
        """
        Extracts face patches from the image based on the provided bounding boxes.

        Args:
            image (numpy.ndarray): The input image.
            bounding_boxes (list): A list of bounding boxes [x1, y1, x2, y2].

        Returns:
            list: A list of extracted face patches.
        """

        patches = []
        for box in bounding_boxes:
            x1, y1, x2, y2 = box
            x1, y1 = max(0, x1), max(0, y1) # Clip coordinates to image boundaries
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            face_patch = image[y1:y2, x1:x2]
            patches.append(face_patch)
        return patches
