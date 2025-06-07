import cv2
import numpy as np
import torch
import torchvision.ops as ops

class FaceExtractions:
    """
    This class extracts face patches from images using a face detector.
    """

    def __init__(self, YOLO_model):
        self.face_detector = YOLO_model  # e.g., a YOLO-based detector

    def extract_patches(self, image:np.ndarray)-> list:
        """
        Extracts face patches from the given image using the face detector.

        Args:
            image (numpy.ndarray): The input image from which to extract faces.

        Returns:
            list: A list of extracted face patches (numpy arrays).
        """

        bb, confidence = self.detect_faces(image)
        patches = self.get_patches(image, bb)
        return patches, confidence
    
    def detect_faces(self, image, threshold=0.5):
        output = self.face_detector(image)  # TODO: Valudate input size: [1, 3, 640, 640]
        output = output.squeeze(0).permute(1,0) # [1, 5, 8400] > [5, 8400] > [8400, 5]
        bboxes, confidence = output[:, :4], output[:, 4]
        
        mask = confidence > threshold
        filtered_bboxes = bboxes[mask]
        filtered_confidences = confidence[mask]

        boxes_xyxy = self.get_xxyyy(filtered_bboxes)

        # Run NMS
        nms_threshold = 0.4  # adjust as needed
        indices = ops.nms(boxes_xyxy, filtered_confidences, nms_threshold)

        return filtered_bboxes[indices], filtered_confidences[indices]
    
    def get_xxyyy(self, boxes):
        x_c, y_c, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_c - w / 2
        y1 = y_c - h / 2
        x2 = x_c + w / 2
        y2 = y_c + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)

    def get_patches(self, image, boxes):
        """
        Extracts face patches from the image based on the provided bounding boxes.

        Args:
            image (numpy.ndarray): The input image.
            bounding_boxes (list): A list of bounding boxes [x1, y1, x2, y2].

        Returns:
            list: A list of extracted face patches.
        """

        patches = []
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()

            # Clip to image boundaries
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(image.shape[1], int(x2))
            y2 = min(image.shape[0], int(y2))

            if x2 > x1 and y2 > y1:
                face_patch = image[y1:y2, x1:x2]
                patches.append(face_patch)

        return patches