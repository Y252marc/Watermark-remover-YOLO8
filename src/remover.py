import cv2
import numpy as np

class WatermarkRemover:
    def __init__(self, method='telea'):
        """
        Initialize the remover. 
        method: 'telea' (good for small details) or 'ns' (Navier-Stokes, good for larger areas)
        """
        self.method = cv2.INPAINT_TELEA if method == 'telea' else cv2.INPAINT_NS

    def remove(self, frame, boxes):
        """
        Takes a frame and a list of bounding boxes [(x1, y1, x2, y2)...]
        Returns the frame with those areas inpainted (removed).
        """
        if not boxes:
            return frame

        # 1. Create a Mask
        # A mask is a black image with the same dimensions as the frame
        # uint8 is the standard image type (0-255)
        mask = np.zeros(frame.shape[:2], dtype="uint8")

        # 2. Draw the 'damage' on the mask
        for (x1, y1, x2, y2) in boxes:
            # We expand the box slightly (padding) to ensure we cover the edges
            padding = 5
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)

            # Draw a white rectangle on the mask where the object is
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        # 3. Apply Inpainting
        # The '3' is the radius (how far to look for neighbor pixels)
        cleaned_frame = cv2.inpaint(frame, mask, 3, self.method)

        return cleaned_frame