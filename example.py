from UWEC_CP.utils import Config, yolo_v2
from UWEC_CP import Parser



config = Config('curling1.mp4', yolo_v2, visualize=False, save_interval=50, yolo_log=False)

parser = Parser(config)

parser.parse_video()
