import os

class Config:
    _class_list = ['BroomHead', 'Hack', 'Hogline', 'House', 'Player', 'Rock'] 
    _field_names = ['frame', 'object_class', 'id', 'box_coords', 'center', 'color', 'radius']
    
    def __init__(
        self, 
        video: str, #path to video/link to video
        model: str, #path to model
        conf_threshold: float = 0.7, #determines the confidence
        save_interval: int = 0, #determines how many frames before the parser will append the csv file
        data_dir: str = 'parser_data', #determines where the data will be stored
        data_file_name: str = 'data', #determines the name of the data file
        is_yt: bool = False, #determines if video will be treated as a path or link
        visualize: bool = False, #determines if the model will produce an image; for debugging
        yolo_log: bool = False, #determines if the model will output to the terminal; for debugging
        ) -> None:
        
        assert conf_threshold >= 0 and conf_threshold <= 1, "The confidence threshold needs to be between 0 and 1"
        assert save_interval >= 0, "The save interval must be greater than 0 or zero to disable saving while parsing"
        
        if not is_yt:
            if not os.path.exists(video):
                raise FileNotFoundError(f"Video path not found: {video}")
        else:
            #check if link works
            pass
            
        if not os.path.exists(model):
            raise FileNotFoundError(f"Video path not found: {model}")
        
        #if the data file name does not have a .csv extension then add the extension
        original_name = data_file_name
        root, ext = os.path.splitext(original_name)
        if ext.lower() != '.csv':
            data_file_name = f"{root}.csv"
        
        if os.path.exists(os.path.join(data_dir, data_file_name)):
            print(f"Data file already exists: {os.path.join(data_dir, data_file_name)}")
            awnser = input("Would you like to overwrite the file (y/n): ")
            if awnser == 'y' or awnser == '':
                pass
            elif awnser == 'n':
                print("Please use a different Data File name")
                quit()
            else: 
                print("Invalid Response")
                quit()
        
        self.video = video
        self.model = model
        self.conf_threshold = conf_threshold
        self.save_interval = save_interval
        self.data_dir = data_dir
        self.data_file_name = data_file_name
        self.is_yt = is_yt
        self.visualize = visualize
        self.yolo_log = yolo_log
        
            
    def __str__(self) -> str:
        return f'''
            Your Config
            Vido Path: {self.video}
            Model: {self.model}
            Confidence Threshold: {self.conf_threshold}
            Save Interval: {self.save_interval}
            Data Directory: {self.data_dir}
            Data File Name: {self.data_file_name}
            Is Youtube: {self.is_yt}
            Visualiz: {self.visualize}
            Yolo Log: {self.yolo_log}
            '''