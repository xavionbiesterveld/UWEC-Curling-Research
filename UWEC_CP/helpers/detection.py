import math
import cv2
import numpy as np
import statistics
from ultralytics.engine.results import Boxes
from typing import Dict

def get_detection_properties(box: Boxes) -> Dict[str, float | int | tuple[int, int, int, int]]:
    """Convert a single YOLO detection from a Boxes object into a Python dictionary.

    Args:
        box (ultralytics.engine.results.Boxes): A single detection object from Ultralytics YOLO,
            containing tensor attributes like cls, conf, id, and xyxy.

    Returns:
        dict: A dictionary with detection properties:
            - 'object_class' (int): The class ID of the detected object.
            - 'confidence' (float): Confidence score, rounded to 2 decimal places.
            - 'id' (int): Unique identifier of the detection.
            - 'coordinates' (tuple[int, int, int, int]): Bounding box coordinates (x1, y1, x2, y2).
            
    Raises:
        ValueError: If the Boxes object is invalid or missing required attributes.
    """
    detection_dict = {}
    try:
        #extract detection data from tensor and put into dictionary
        detection_dict.update({
            'object_class': int(box.cls.numpy()[0]),
            'confidence': math.ceil(box.conf.numpy()[0] * 100) / 100,
            'id': int(box.id.numpy()[0]),
            'coordinates': tuple(map(int, box.xyxy.numpy()[0]))
        })
    except (AttributeError, IndexError, ValueError) as e:
        raise ValueError(f"Invalid Boxes object: {e}") from e

    return detection_dict

def get_calculated_detection_properties(detection_dict: Dict[str, float | int | tuple[int, int, int, int]]) -> Dict[str, float | int | tuple[int, int, int, int] | tuple[int, int] | None]:
    """Calculate additional properties from a detection dictionary and append them.

    Args:
        detection_dict (dict): A dictionary containing detection properties from a YOLO Boxes object.
            Expected keys: 'object_class' (int), 'confidence' (float), 'id' (int), 
            'coordinates' (tuple[int, int, int, int] for x1, y1, x2, y2).

    Returns:
        dict: The input dictionary updated with:
            - 'center' (tuple[int, int]): Coordinates of the bounding box center (x, y).
            - 'radius' (int or None): Half the height of the box, or None if object_class is 4 (a person). Assumes box is close to the shape of a square.
            
    Raises:
        KeyError: If required keys (e.g., 'coordinates', 'object_class') are missing.
        TypeError: If 'coordinates' is not a tuple of four integers or other values have incorrect types.
    """

    # Validate required keys
    required_keys = {'coordinates', 'object_class'}
    missing_keys = required_keys - set(detection_dict.keys())
    if missing_keys:
        raise KeyError(f"Missing required keys: {missing_keys}")
    
    # Validate coordinates
    coords = detection_dict['coordinates']
    if not (isinstance(coords, tuple) and len(coords) == 4 and all(isinstance(x, int) for x in coords)):
        raise TypeError("Coordinates must be a tuple of four integers (x1, y1, x2, y2)")
    
    x1, y1, x2, y2 = coords
    
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    
    # Calculate radius unless object_class is a non-circular object
    object_class = detection_dict['object_class']
    NON_CIRCULAR_CLASSES = [4]
    radius = int((y2 - y1) / 2) if object_class not in NON_CIRCULAR_CLASSES else None
    
    detection_dict.update({
        'center': center,
        'radius': radius
    })
    
    return detection_dict


def visualize_box(img: np.ndarray, detection_dict: Dict[str, float | int | tuple[int, int, int, int] | tuple[int, int] | None], show_circle: bool = True) -> np.ndarray:
    """Visualize a YOLO detection on a BGR image using bounding box and optional circle annotations.

    This function draws a red bounding box around a detected object and, if specified, a green circle
    centered at the detection's center with a radius (for circular objects). It modifies the input
    image in-place and returns it for further use. 
    
    Args:
        img (np.ndarray): A BGR image (height, width, 3) with dtype uint8, typically from cv2.imread(). 
        detection_dict (dict): A dictionary containing detection properties from a YOLO Boxes object.
           - 'coordinates' (tuple[int, int, int, int]): Bounding box coordinates (x1, y1, x2, y2).
           - 'center' (tuple[int, int]): Coordinates of the bounding box center (x, y)
           - 'radius' (int or None): Half the height of the box, or None if object is not circular.
        show_circle (bool, optional): If true, visualizes a circle on the image using the radius of the detection if it exists. Defaults to True.

    Returns:
        np.ndarray: The input image with a red bounding box and, if applicable, a green circle drawn on it.
    
    Raises: 
        KeyError: If required keys (e.g., 'coordinates', 'radius', 'center') are missing.
        ValueError: If the radius is neither an integer nor None.
        ValueError: If 'img' is not a valid BGR image (wrong shape or dtype).
    """
    # Validate image
    if not (isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8):
        raise ValueError("Image must be a BGR numpy array with shape (height, width, 3) and dtype uint8")
    
    # Validate required keys
    required_keys = {'coordinates', 'center', 'radius'}
    missing_keys = required_keys - set(detection_dict.keys())
    if missing_keys:
        raise KeyError(f"Missing required keys: {missing_keys}")
    
    # Visualize red rectange
    x1, y1, x2, y2 = detection_dict['coordinates']
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
    
    # Visualize green circle if radius exists and show circle is True
    radius = detection_dict['radius']
    if not (isinstance(radius, (int, type(None)))):
        raise TypeError(f"The radius must be an integer or None")
    
    if show_circle and radius:
        cv2.circle(img, detection_dict['center'], radius, (0, 255, 0), 2)
        
    return img

def new_find_color(img: np.ndarray, detection_dict: Dict[str, float | int | tuple[int, int, int, int] | tuple[int, int] | None], sample_number: int = 50) -> str:   
    """Determine whether the color of a detection is red or yellow.
    
    This function samples an inputed amount of pixels from a detection and calculates
    the distance between the BGR values a dictionary of colors and the median sample pixel.
    
    Args:
        img (np.ndarray): A BGR image (height, width, 3) with dtype uint8, typically from cv2.imread(). 
        detection_dict (dict): A dictionary containing detection properties from a YOLO Boxes object.
           - 'center' (tuple[int, int]): Coordinates of the bounding box center (x, y).
           - 'radius' (int or None): Half the height of the box, or None if object is not circular.
        sample_number (int): The number of sample pixels that will be taken from the detection.
        
    Returns:
        str: A string containing the name of the color detected.
        
    Raises:
        AssertionError: If the sample number is less than zero or greater than the amount of pixels in the image.
        TypeError: The radius must be an integer. 
        ValueError: Image must be a BGR numpy array with shape (height, width, 3) and dtype uint8.
    """ 
    
    if not (isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8):
        raise ValueError("Image must be a BGR numpy array with shape (height, width, 3) and dtype uint8")
    
    radius = detection_dict['radius']
    if not (isinstance(radius)):
        raise TypeError(f"The radius must be an integer")
    
    #find height and width of the image
    height, width = img.shape[:2]
    
    assert sample_number > 0 and sample_number < (height * width), "The sample size must be more than zero and less than the amount of pixels in the image"
    
    #create a mask with the size of the image and draw a filled in circle onto it
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, detection_dict['center'], radius, 255, -1)
    
    #find all of the points of the circle
    points = np.argwhere(mask == 255)
    
    #if the sample number is larger than the amount of pixels in the circle then use the amount of pixels in the circle
    n_samples = min(sample_number, len(points))
    
    #select n sample pixels from circle
    selected_indices = np.random.choice(len(points), size=n_samples, replace=False)
    selected_points = points[selected_indices]
    pixels = img[selected_points[:, 0], selected_points[:, 1]]
    
    #find the median of the color values from the list of sample pixels
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
    
    #find the euclidean distance between the median pixel and the colors in the dictionary
    distances = {}
    for key, value in colors.items():
        distance = sum((int(a) - int(b)) ** 2 for a, b in zip(avg_pixel, value)) ** 0.5
        
        distances.update({key: distance})
    
    #select the color closest to the median pixel
    closest_color = min(distances, key=distances.get)
    return closest_color

def resize_image(img: np.ndarray, target_resolution: tuple[int, int]) -> np.ndarray:
    """Resizes an input image to a target resolution.

    Args:
        img (np.ndarray): A BGR image (height, width, 3) with dtype uint8, typically from cv2.imread(). 
        target_resolution (tuple[int, int]): A tuple containing the height and width of the desired resolution. 

    Returns:
        np.ndarray: The image resized to the target resolution.
        
    Raises:
        ValueError: Image must be a BGR numpy array with shape (height, width, 3) and dtype uint8.
        TypeError: Resolution tuple must be a tuple of two integers.
        AssertionError: The target height and width of the resolution must be greater than zero.
    """
    
    if not (isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8):
        raise ValueError("Image must be a BGR numpy array with shape (height, width, 3) and dtype uint8")
    
    if not (isinstance(target_resolution, tuple) and len(target_resolution) == 2 and all(isinstance(x, int) for x in target_resolution)):
        raise TypeError("Target resolution must be a tuple of two integers (height, width)")

    #takes the img and target_resolution as a tuple ex: (1920, 1080)
    h, w = img.shape[:2]
    target_w, target_h = target_resolution
    
    assert target_w > 0 and target_h > 0, 'The height and width of the resoultion must be greater than zero'
    
    # Calculate scale factors for both dimensions
    scale_x = target_w / w
    scale_y = target_h / h
    scale = min(scale_x, scale_y)  # Maintain aspect ratio
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    return cv2.resize(img, (new_w, new_h), interpolation=1)