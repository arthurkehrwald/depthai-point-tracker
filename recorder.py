import csv
import os
from datetime import datetime

class Recorder:
    def __init__(self, directory="recordings"):
        self.directory = directory
        self.file = None
        self.writer = None
        self.filepath = None

    def start(self):
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.csv")
        self.filepath = os.path.join(self.directory, filename)
        
        self.file = open(self.filepath, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['x', 'y', 'z', 'capture_time', 'detection_time', 'confidence'])
        return self.filepath

    def record(self, x, y, z, capture_time, detection_time, confidence):
        if self.writer:
            self.writer.writerow([x, y, z, capture_time, detection_time, confidence])

    def stop(self):
        if self.file:
            self.file.close()
            self.file = None
            self.writer = None
            self.filepath = None
