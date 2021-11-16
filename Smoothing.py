import numpy as np

class Smoothing:
    def __init__(self, num_samples=10) -> None:
        self.num_samples = num_samples
        self.count = 0
        self.currentSample = None
        self.lastSample = None

        self.currentValue = None

        self.samples = []


    def getSample(self, sample):
        if self.num_samples == 1:
            return sample

        self.count += 1
        
        if self.currentSample is None:
            self.currentSample = sample
            self.lastSample = sample

        if self.currentValue is None:
            self.currentValue = sample

        if self.count < self.num_samples:
            self.samples.append(sample)

            self.currentValue = self.lastSample + self.count*((self.currentSample-self.lastSample)/self.num_samples)

        else:
            self.lastSample = self.currentSample
            self.currentSample = np.average(self.samples, axis=0)
            self.samples.clear()
            self.count = 0

        return self.currentValue
