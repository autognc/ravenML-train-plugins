import numpy as np

class DetectedClass():
    def __init__(self, class_id, class_name, score, box, mask):
        self.class_id = class_id
        self.class_name = class_name
        self.score = score
        self.box = box
        self.mask = mask
        
        self.calculate_centroid()
        
    def calculate_centroid(self):
        idxs = np.where(self.mask == 1)
        if len(idxs[0]) > 0 and len(idxs[1]) > 0:
            self.centroid = (int(round(np.average(idxs[0]))), int(round(np.average(idxs[1]))))
        else:
            self.centroid = None


class TruthClass():
    def __init__(self, class_id, class_name, mask, centroid):
        self.class_id = class_id
        self.class_name = class_name
        self.mask = mask
        self.centroid = centroid