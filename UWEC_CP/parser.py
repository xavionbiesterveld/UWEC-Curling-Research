import logging
import cv2
from ultralytics import YOLO
from tqdm import tqdm

from .helpers.data import *
from .helpers.detection import *
from .assets import mask_a

class Parser:
    
    def __init__(self, config) -> None:
        self.config = config
    
    def parse_video(self) -> None:
        video = cv2.VideoCapture(self.config.video)
        model = YOLO(self.config.model)
        mask = cv2.imread(mask_a)
        
        if not self.config.yolo_log:
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            progress = tqdm(total=total_frames, desc='Processing video', unit='frame')
        
        if not self.config.yolo_log:
            logging.getLogger("ultralytics").setLevel(logging.ERROR)
        
        video_unfinished = True
        frame_number = 1
        frames_since_save = 0
        data = []
        
        new_path = initialize_csv(self.config.data_dir, self.config.data_file_name, self.config._field_names)
        
        while video_unfinished:
            video_unfinished, img = video.read()
            
            if not video_unfinished:
                break
            
            if not self.config.yolo_log:
                progress.update(1)
            
            imgMask = cv2.bitwise_and(img, mask)
            results = model.track(imgMask, stream=True, persist=True)
            
            for result in results:
                for box in result.boxes:
                    box_properties = get_detection_properties(box)
                    
                    if box_properties['object_class'] == 5 and box_properties['confidence'] >= self.config.conf_threshold:
                        box_properties = get_calculated_detection_properties(box_properties)
                        
                        
                        color = new_find_color(img, box_properties['center'], int(box_properties['radius']), 50)
                        
                        data.append({
                        'frame': frame_number,
                        'object_class': self.config._class_list[box_properties['object_class']],
                        'id': box_properties['id'],
                        'box_coords': box_properties['coordinates'],
                        'center': box_properties['center'],
                        'color': color,
                        'radius': box_properties['radius']
                        })
                    
                        if self.config.visualize:
                            imgMask = visualize_box(imgMask, tuple(box_properties['coordinates']), box_properties['center'], box_properties['radius'])
                            cv2.imshow('Img', imgMask)
                            cv2.waitKey(0)
             
            if self.config.save_interval > 0: 
                frames_since_save += 1
                
                #if the save interval is reached then the data file will be appended.
                if frames_since_save >= self.config.save_interval and data:   
                    append_csv(data, new_path, self.config._field_names)
                    
                    data.clear()
                    frames_since_save = 0
                    
            frame_number += 1
        if not self.config.yolo_log:
            progress.close() 
        video.release()
        
        if data:
            append_csv(data, new_path, self.config._field_names)