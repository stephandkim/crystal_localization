import numpy as np
# import cupy as np
import cv2
import matplotlib.pyplot as plt
import queue
import src.constants as constants

class ImageObject(object):
    
    def __init__(self, raw):
        self.raw = raw
        self.vectorized = np.float32(self.raw.reshape((-1,3)))
        
        self.clustered = None
        self.background_idx = None
        
        self.map = None
        
        self.num_polygons = None
        
        
    def prepare_map(self, idx_background):
#     Distinguish the background tiles from polygons. Assign uncounted values for polygon tiles.

        self.map = np.zeros(self.clustered.shape[0:2])

        for y in range(self.clustered.shape[0]):
            for x in range(self.clustered.shape[1]):
                if all(self.clustered[y, x] != self.center[idx_background]):
                    self.map[y, x] = constants.MAP_CODE['uncounted']
                    
                    
                    
    def plot_map(self):
        plot = np.zeros(self.clustered.shape)
        
        for y in range(self.map.shape[0]):
            for x in range(self.map.shape[1]):
                if self.map[y, x] == constants.MAP_CODE['background']:
                    plot[y, x] = [0, 0, 0]
                elif self.map[y, x] == constants.MAP_CODE['enclosed']:
                    plot[y, x] = [1, 0, 0]
                elif self.map[y, x] == constants.MAP_CODE['unenclosed']:
                    plot[y, x] = [0, 0, 1]
                else:
                    plot[y, x] = [1, 1, 1]
                    
        plt.imshow(plot)