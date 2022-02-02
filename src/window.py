import math
from utils import *
from src import *
from collections import namedtuple

Coordinates = namedtuple('Coordinates', ('ymin', 'ymax', 'xmin', 'xmax'))
Square = namedtuple('Square', ('m', 'M'))

class Polygon(object):
    
    def __init__(self, num_windows):
        self.pixel_location_info = np.zeros((num_windows,))
        
class Window(object):
    
    def __init__(self, square=Square(1,1), coordinates=Coordinates(0,1,0,1)):
        self.coordinates = coordinates
        self.square = square
        
        self.polygons_enclosed = 0
        self.polygons_unenclosed = 0
        
        self.pixel_ratio = 0
        self.r_pixel = 0
        self.r_polygons = 0
        self.r_square = 0
        self.r_total = 0
        
        

#     def clear_unenclosed_polygons(self):
#         for y in range(self.local_map.shape[0]):
#             for x in range(self.local_map.shape[1]):
#                 if self.local_map[y, x] == constants.MAP_CODE['unenclosed']:
#                     self.local_map = unfill_unenclosed(self.local_map, y, x)
                    
#     def clear_enclosed_polygons(self):
#         for y in range(self.local_map.shape[0]):
#             for x in range(self.local_map.shape[1]):
#                 if self.local_map[y, x] == constants.MAP_CODE['enclosed']:
#                     self.local_map = unfill_enclosed(self.local_map, y, x)



def get_windows(M, coordinates):
#     Takes the coordinates of the largest window, L, and M and return subwindows.
    
    windows = []
    windows_hierarchy = []
    order = 1
    
    dy = (coordinates.ymax - coordinates.ymin)/M
    dx = (coordinates.xmax - coordinates.xmin)/M
    
    for m in range(1, M+1):
        for l in range(1, M+1):
            if m == l:
                for k in range(M-m+1):
                    for j in range(M-m+1):
                        
                        y0 = coordinates.ymin + k*dy
                        y1 = y0 + m*dy

                        x0 = coordinates.xmin + j*dx
                        x1 = x0 + l*dx

                        windows.append(Window(
                            Square(int(m), int(M)),
                            Coordinates(int(y0), int(y1), int(x0), int(x1))
                        ))
                        windows_hierarchy.append(M-order)
                        
        
        order += 1
    return windows, windows_hierarchy


def flood_fill(img_map, y, x, windows):
    if img_map is None:
        print('The img map is not ready. \n')
        return
    
    if ((y >= img_map.shape[0])
        or (y < 0)
        or (x >= img_map.shape[1])
        or (x < 0)
        or (img_map[y, x] != constants.MAP_CODE['uncounted'])
       ):
        
        print('Selection error.')
        return

    enclosed_queue = queue.Queue()
    unenclosed_queue = queue.Queue()
    enclosed = True
    polygon = Polygon(len(windows))
    global_coordinates = windows[-1].coordinates
    
    enclosed_queue.put([y, x])
    while ((not enclosed_queue.empty()) or (not unenclosed_queue.empty())):
        y, x = enclosed_queue.get() if (not enclosed_queue.empty()) else unenclosed_queue.get()
        if ((y >= img_map.shape[0])
            or (y < 0)
            or (x >= img_map.shape[1])
            or (x < 0)):
            enclosed = False
            continue
        
        if enclosed:
            if ((img_map[y, x] == constants.MAP_CODE['background']) 
                or (img_map[y, x] == constants.MAP_CODE['enclosed'])
               ):
                continue
            else:
                img_map[y, x] = constants.MAP_CODE['enclosed']
                
                for idx, window in enumerate(windows):
                    if ((y+global_coordinates.ymin>window.coordinates.ymin)
                        and (y+global_coordinates.ymin<window.coordinates.ymax)
                        and (x+global_coordinates.xmin>window.coordinates.xmin)
                        and (x+global_coordinates.xmin<window.coordinates.xmax)
                    ):
                        polygon.pixel_location_info[idx] += 1
                        
                enclosed_queue.put([y+1, x])
                enclosed_queue.put([y-1, x])
                enclosed_queue.put([y, x+1])
                enclosed_queue.put([y, x-1])
        else:
            enclosed_queue.queue.clear()
            polygon = None
            
            if ((img_map[y, x] == constants.MAP_CODE['background']) 
                or (img_map[y, x] == constants.MAP_CODE['unenclosed'])
               ):
                continue
            else:
                img_map[y, x] = constants.MAP_CODE['unenclosed']
                unenclosed_queue.put([y+1, x])
                unenclosed_queue.put([y-1, x])
                unenclosed_queue.put([y, x+1])
                unenclosed_queue.put([y, x-1])
            
            
    return img_map, polygon, enclosed
