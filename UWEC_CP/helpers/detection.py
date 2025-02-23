import math
import cv2
import numpy as np
import statistics

def get_detection_properties(box) -> dict:
    #extract information from box object, a collection of tensors that hold information about a box_properties
    #return a dictionary of information about a box_properties
    box_properties = {}

    #.numpy()[0] => converts gpu tensor into numpy array and picks the first element.
    object_class = int(box.cls.numpy()[0])
    confidence = math.ceil(box.conf.numpy()[0] * 100) / 100
    box_id = int(box.id.numpy()[0])
    x1, y1, x2, y2 = map(int, box.xyxy.numpy()[0])

    box_properties.update({
        'object_class': object_class,
        'confidence': confidence,
        'id': box_id,
        'coordinates': (x1, y1, x2, y2)
    })

    return box_properties

def get_calculated_detection_properties(box_properties: dict) -> dict:

    x1, y1, x2, y2 = box_properties['coordinates']
    
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    radius = (y2 - y1) / 2
    
    box_properties.update({
        'center': center,
        'radius': radius
    })
    
    return box_properties


def visualize_box(img, coordinates, center, radius, show_circ: bool = True):
    #visualize detections by placing bounding box on image
    x1, y1, x2, y2 = coordinates
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
    
    if show_circ:
        cv2.circle(img, center, math.floor(radius), (0, 255, 0), 2)
        
    return img

def new_find_color(img, center, radius, n):    
    height, width = img.shape[:2]
    
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    
    points = np.argwhere(mask == 255)
    
    n_samples = min(n, len(points))
    selected_indices = np.random.choice(len(points), size=n_samples, replace=False)
    selected_points = points[selected_indices]
    
    pixels = img[selected_points[:, 0], selected_points[:, 1]]
    
    
    pixels_int = [(int(b), int(g), int(r)) for b, g, r in pixels]
    zip_channels = list(zip(*pixels_int))
    avg_pixel = (
        statistics.median(zip_channels[0]),
        statistics.median(zip_channels[1]),
        statistics.median(zip_channels[2])
    )
    
    colors = {
        'red': (0, 0, 255),
        'yellow': (0, 255, 255)
    }
    
    distances = {}
    
    for key, value in colors.items():
        distance = sum((int(a) - int(b)) ** 2 for a, b in zip(avg_pixel, value)) ** 0.5
        
        distances.update({key: distance})
        
    closest_color = min(distances, key=distances.get)
    return closest_color

def resize_image(img, target_resolution):
    #takes the img and target_resolution as a tuple ex: (1920, 1080)
    h, w = img.shape[:2]
    target_w, target_h = target_resolution
    
    # Calculate scale factors for both dimensions
    scale_x = target_w / w
    scale_y = target_h / h
    scale = min(scale_x, scale_y)  # Maintain aspect ratio
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    return cv2.resize(img, (new_w, new_h), interpolation=1)