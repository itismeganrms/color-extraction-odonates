import cv2
import os
import numpy as np

def draw_polygons(image_path, label_path, class_colors=None):
    # Load image (just for visualization, could also use a blank canvas)
    image = cv2.imread(image_path)
    H, W, _ = image.shape

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        values = line.strip().split()
        cls = int(values[0])  # class_id
        coords = list(map(float, values[1:]))  # polygon coords

        # Group into (x, y) pairs and denormalize
        polygon = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * W)
            y = int(coords[i+1] * H)
            polygon.append([x, y])
        polygon = np.array(polygon, dtype=np.int32)

        # Pick color (per class or random if not provided)
        color = class_colors[cls] if class_colors else (0, 255, 0)

        # Draw polygon outline
        cv2.polylines(image, [polygon], isClosed=True, color=color, thickness=2)

        # Optionally fill polygon
        # cv2.fillPoly(image, [polygon], color)

        # Put class id
        cv2.putText(image, str(cls), tuple(polygon[0]), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, color, 2)

    return image


# Example usage:
image_path = "/home/mrajaraman/master-thesis-dragonfly/yolo-model/batch_annotations/first_batch/img_5179872445.jpg"
label_path = "/home/mrajaraman/master-thesis-dragonfly/yolo-model/batch_annotations/img_5179872445.txt"
vis = draw_polygons(image_path, label_path)

cv2.imwrite("vis_output_new_batch.png", vis)


# import cv2
# import numpy as np

# # Example image size
# W, H = 512, 512

# # Normalized coordinates from your example
# coords = [0.0, 0.0, 0.0, 0.99875, 0.998708, 0.99875, 0.998708, 0.0]

# # Convert to pixel coordinates
# polygon = np.array([[int(x * W), int(y * H)] for x, y in zip(coords[::2], coords[1::2])], dtype=np.int32)

# # Create blank image
# img = np.zeros((H, W, 3), dtype=np.uint8)

# # Draw the polygon
# cv2.polylines(img, [polygon], isClosed=True, color=(0,255,0), thickness=2)

# # Fill the polygon (optional)
# # cv2.fillPoly(img, [polygon], color=(0,255,0))

# # Show the image
# cv2.imwrite("polygon.png", img)
