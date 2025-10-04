import numpy as np

class BoundingBox:
    """
    For 3D BoundingBox
    """
    def __init__(self):
        self.clear()
    
    def clear(self):
        """
        Intialize min and max as opposite infinite value
        """
        self.__somePrivate__ = ""
        self._min = np.array([np.inf] * 3, dtype=np.float64) 
        self._max = np.array([-np.inf] * 3, dtype=np.float64)

    @property
    def min(self):
        return self._min 
    @min.setter
    def min(self, value):
        self._min = value
    
    @property
    def max(self):
        return self._max
    @max.setter
    def max(self, value):
        self._max = value

    @property
    def center(self):
        return (self.min + self.max) * 0.5

    @property
    def diagonal(self):
        return np.linalg.norm(self.min - self.max)

