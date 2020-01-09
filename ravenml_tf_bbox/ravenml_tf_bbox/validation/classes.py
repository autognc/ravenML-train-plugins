class DetectedClass:
    def __init__(self, class_name, score, box):
        self.class_name = class_name
        self.score = score
        self.box = box


class TruthClass:
    def __init__(self, class_name, box, xdim, ydim):
        self.class_name = class_name
        self.box = box
        self.xdim = xdim
        self.ydim = ydim

        self.box_norm = {
            'xmin': self.box['xmin'] / self.xdim,
            'xmax': self.box['xmax'] / self.xdim,
            'ymin': self.box['ymin'] / self.ydim,
            'ymax': self.box['ymax'] / self.ydim
        }

    def __repr__(self):
        return self.class_name + ': ' + str(self.box_norm)
