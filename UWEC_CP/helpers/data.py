import os
import csv

def initialize_csv(data_dir: str, data_file: str, field_names: list) -> str:
    # Create log directory and CSV file with headers
    os.makedirs(data_dir, exist_ok=True)
    new_dir = os.path.join(data_dir, data_file)
    
    with open(new_dir, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        
    return new_dir
        
    

def append_csv(data: list, data_path: str, field_names: list) -> None:
    # Append CSV file
    with open(data_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writerows(data)