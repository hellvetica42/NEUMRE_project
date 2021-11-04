import numpy as np

class Smoothing:
    def __init__(self, sample_size=10) -> None:
        self.sample_size = sample_size
        self.current_kps = None
        self.accumulator = None
        self.sample_count = 0

    def add_sample(self, keypoints):
        if self.current_kps is None:
            self.current_kps = keypoints

        if self.accumulator is None:
            self.accumulator = np.zeros(keypoints.shape)

        if self.sample_count < self.sample_size:
            self.accumulator += keypoints
            self.sample_count += 1
        
        else:
            self.accumulator /= self.sample_size
            self.current_kps = self.accumulator
            self.accumulator = None
            self.sample_count = 0

        return self.current_kps
