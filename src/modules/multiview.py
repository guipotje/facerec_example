import cv2
import numpy as np

class Validator:
    """
    Class that generates random ellipsoids where the user has to position its face to validate it.
    """
    def __init__(self):
        self.min_radius = 0.26  # Minimum radius as a fraction of the image height
        self.max_radius = 0.32  # Maximum radius as a fraction of the image height
        self.max_offset = 0.15  # Maximum offset as a fraction of the image dimensions

    def gen_random_ellipsoid(self, image_shape):
        """
        Generate a random ellipsoid where the major axis (always vertical) is within min and max radius,
        and the minor axis is half of the major axis. The center of the ellipsoid is always centered in the image
        with a random offset (max offset) added.
        """
        height, width = image_shape[:2]
        max_dim = min(height, width)
        
        major_axis = np.random.uniform(self.min_radius, self.max_radius) * max_dim
        minor_axis = major_axis / 1.5
        
        center_x = width // 2 + int((np.random.uniform(-1, 1) * self.max_offset) * width)
        center_y = height // 2 + int((np.random.uniform(-1, 1) * self.max_offset) * height)

        return (center_x, center_y), (int(minor_axis), int(major_axis))

    def check_inside(self, mask, bbox):
        """
        Check if bbox is inside the mask. Returns True if at least x% of bbox area is inside the mask.
        """
        x, y, w, h = bbox
        roi = mask[y:y+h, x:x+w]
        non_zero_count = np.count_nonzero(roi)
        bbox_area = w * h
        print(non_zero_count / bbox_area , flush=True)
        return non_zero_count / bbox_area >= 0.85

    def draw_ellipsoid(self, image, center, axes, alpha=0.6):
        """
        Draw a semi-transparent ellipsoid mask with alpha shading on the given image.
        The ellipsoid area is transparent, while the rest of the image is shaded.
        Uses anti-aliasing for smoother edges.
        """
        mask = np.ones_like(image, dtype=np.uint8) * 255  # White mask
        cv2.ellipse(mask, center, axes, 0, 0, 360, (0, 0, 0), -1)  # Black ellipsoid with anti-aliasing
        mask_ellips = mask[:, :, 0] == 0
        mask[mask_ellips] = image[mask_ellips]

        overlay = image.copy()
        shaded_image = cv2.addWeighted(overlay, 1 - alpha, mask, alpha, 0)

        return shaded_image, mask_ellips  # Return the shaded image and the mask
