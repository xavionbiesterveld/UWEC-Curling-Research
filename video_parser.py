import cv2
from ultralytics import YOLO
import math
import webcolors
import os
import csv

CLASS_LIST = ['BroomHead', 'Hack', 'Hogline', 'House', 'Player', 'Rock'] 
VIDEO_PATH = os.path.join('video', 'curlingvideo.mp4')
MODEL_PATH = os.path.join('model_v2', 'runs', 'train2', 'weights', 'best.pt')
CONF_THRESHOLD = 0.63
SAVE_INTERVAL = 100
DATA_DIR = 'parser_data'
DATA_FILE = os.path.join(DATA_DIR, f'{os.path.splitext(os.path.basename(VIDEO_PATH))[0]}_frame_data.csv')
FIELD_NAMES = ['frame', 'object_class', 'id', 'box_coords', 'center', 'color', 'radius']


def get_detection_properties(box):
    #extract information from box object, a collection of tensors that hold information about a box_properties
    #return a dictionary of information about a box_properties
    box_properties = {}

    #.numpy()[0] => converts gpu tensor into numpy array and picks the first element.
    object_class = int(box.cls.numpy()[0])
    confidence = math.ceil(box.conf.numpy()[0] * 100) / 100
    box_id = int(box.id.numpy()[0])
    x1, y1, x2, y2 = map(int, box.xyxy.numpy()[0])

    #find calculated values
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    radius = (y2 - y1) / 2

    box_properties.update({
        'object_class': object_class,
        'confidence': confidence,
        'id': box_id,
        'coordinates': (x1, y1, x2, y2),
        'center': center,
        'radius': radius
    })

    return box_properties

def find_closest_color(RGB_value):
    #use euclidean distance to find the closest color name from CSS3 color names.
    #returns closest color name
    colors = {}

    def euclidean_distance(rgb1, rgb2):
         return sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)) ** 0.5

    
    for name in webcolors.names('html4'):
        r_c, g_c, b_c = webcolors.name_to_rgb(name)
        #values stored in dictionary as {distance: 'color name'}
        colors[(euclidean_distance(RGB_value, (r_c, g_c, b_c)))] = name

    return colors[min(colors.keys())]

def initialize_csv():
    # Create log directory and CSV file with headers
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DATA_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_NAMES)
        writer.writeheader()
        
def validate_paths():
    errors = []
    
    # Validate file paths
    for path, desc in [
        (VIDEO_PATH, "Source video"),
        (MODEL_PATH, "YOLO model")
    ]:
        if not os.path.isfile(path):
            errors.append(f"{desc} path validation failed - file not found: {path}")

    if errors:
        error_msg = "Critical path validation failed:\n" + "\n".join(errors)
        raise FileNotFoundError(error_msg)


validate_paths() 
initialize_csv()

video = cv2.VideoCapture(VIDEO_PATH)
model = YOLO(MODEL_PATH)
mask = cv2.imread(os.path.join('masks', 'mask_a.png'))

video_unfinished = True
frame_number = 1
frames_since_save = 0
data = []

while video_unfinished:
    video_unfinished, img = video.read()
    if not video_unfinished:
        break  # Exit loop immediately if video ends
    
    imgMask = cv2.bitwise_and(img, mask)
    results = model.track(imgMask, stream=True, persist=True)
    
    for result in results:
        for box in result.boxes:
            box_properties = get_detection_properties(box)
            
            #if object is a rock then record its info
            if box_properties['object_class'] == 5 and box_properties['confidence'] >= CONF_THRESHOLD:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                color = find_closest_color(tuple(img_rgb[box_properties['center'][1], box_properties['center'][0]]))
                
                data.append({
                    'frame': frame_number,
                    'object_class': CLASS_LIST[box_properties['object_class']],
                    'id': box_properties['id'],
                    'box_coords': box_properties['coordinates'],
                    'center': box_properties['center'],
                    'color': color,
                    'radius': box_properties['radius']
                })

    
    frames_since_save += 1
    
    #if the save interval is reached then the data file will be appended.
    if frames_since_save >= SAVE_INTERVAL and data:
        with open(DATA_FILE, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELD_NAMES)
            writer.writerows(data)
        data.clear()
        frames_since_save = 0
    
    frame_number += 1
    
video.release() 

# Final write for remaining data after loop exits
if data:
    with open(DATA_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_NAMES)
        writer.writerows(data)

