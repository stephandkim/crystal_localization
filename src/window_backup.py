import math
from utils import *
from src import *
from collections import namedtuple

Coordinates = namedtuple('Coordinates', ('ymin', 'ymax', 'xmin', 'xmax'))
Rectangle = namedtuple('Rectangle', ('m', 'l', 'M', 'L'))

class Window(object):
    
    global_img_map = None
    
    def __init__(self, rectangle=Rectangle(1,1,1,1), coordinates=Coordinates(0,1,0,1)):
        self.coordinates = coordinates
        self.rectangle = rectangle
        
        self.polygons_enclosed = 0
        self.polygons_unenclosed = 0
        
        self.pixel_ratio = 1
        self.r_polygons = 0
        self.r_rectangle = 0
        self.r_total = 0
        
    def reset_map(self):
        self.img_map = Window.global_img_map.copy()

        
    def count_polygons(self, save=False):
#         Calculations can only be done if permanent=True.
        self.polygons_enclosed = 0
        self.polygons_unenclosed = 0
        
        if save:
            bool_map = self.img_map[
                self.coordinates.ymin:self.coordinates.ymax,
                self.coordinates.xmin:self.coordinates.xmax
            ]
        else:
            bool_map = self.img_map[
                self.coordinates.ymin:self.coordinates.ymax,
                self.coordinates.xmin:self.coordinates.xmax
            ].copy()
            
        for y in range(bool_map.shape[0]):
            for x in range(bool_map.shape[1]):
                if bool_map[y, x] == constants.MAP_CODE['uncounted']:
                    bool_map, enclosed = flood_fill(bool_map, y, x)
                    if enclosed:
                        self.polygons_enclosed += 1
                    else:
                        self.polygons_unenclosed += 1
                        
    def clear_unenclosed_polygons(self):
        bool_map = self.img_map[
                self.coordinates.ymin:self.coordinates.ymax,
                self.coordinates.xmin:self.coordinates.xmax
            ]
        for y in range(bool_map.shape[0]):
            for x in range(bool_map.shape[1]):
                if bool_map[y, x] == constants.MAP_CODE['unenclosed']:
                    bool_map = unfill_unenclosed(bool_map, y, x)
                    
    def clear_enclosed_polygons(self):
        bool_map = self.img_map[
                self.coordinates.ymin:self.coordinates.ymax,
                self.coordinates.xmin:self.coordinates.xmax
            ]
        for y in range(bool_map.shape[0]):
            for x in range(bool_map.shape[1]):
                if bool_map[y, x] == constants.MAP_CODE['enclosed']:
                    bool_map = unfill_enclosed(bool_map, y, x)
        
    
    def plot_map(self, ax=None):
        plot = np.zeros((self.img_map.shape[0], self.img_map.shape[1], 3))

        for y in range(self.coordinates.ymin, self.coordinates.ymax):
            for x in range(self.coordinates.xmin, self.coordinates.xmax):
                if self.img_map[y, x] == constants.MAP_CODE['background']:
                    plot[y, x] = [0, 0, 0]
                elif self.img_map[y, x] == constants.MAP_CODE['enclosed']:
                    plot[y, x] = [1, 0, 0]
                elif self.img_map[y, x] == constants.MAP_CODE['unenclosed']:
                    plot[y, x] = [0, 0, 1]
                else:
                    plot[y, x] = [1, 1, 1]
                    
        plot = plot[
            self.coordinates.ymin:self.coordinates.ymax,
            self.coordinates.xmin:self.coordinates.xmax,
        ]
        if ax is None:
            plt.imshow(plot)
        else:
            ax.imshow(plot)
    
    
    def calculate_reward(self):
        bool_map = self.img_map[
                self.coordinates.ymin:self.coordinates.ymax,
                self.coordinates.xmin:self.coordinates.xmax
            ]
        
        pixels_enclosed = 0
        pixels_background = 0
        
        for y in range(bool_map.shape[0]):
            for x in range(bool_map.shape[1]):
                if bool_map[y, x] == constants.MAP_CODE['enclosed']:
                    pixels_enclosed += 1
                elif bool_map[y, x] == constants.MAP_CODE['background']:
                    pixels_background += 1
        
        if self.polygons_enclosed == 0:
            self.pixel_ratio = 100
            self.polygons_enclosed = 100
        else:
            self.pixel_ratio = pixels_background/(pixels_enclosed+1)
            
        self.r_polygons = math.exp(constants.LAMBDA * (1-self.polygons_enclosed * self.pixel_ratio)) 
        self.r_rectangle = math.exp( -1 * constants.ETA * \
                      ( (self.rectangle.L - self.rectangle.l) + (self.rectangle.M - self.rectangle.m)))
        
        self.r_total = self.r_polygons * self.r_rectangle
#         print('pixel: {}, r_poly: {}, r_rect: {}'.format(self.pixel_ratio, r_polygon, r_rectangle))
        return self.r_total




def get_subwindows(M, L, coordinates):
#     Takes the coordinates of the largest window, L, and M and return subwindows.
    
    subwindows = []
    
    dy = (coordinates.ymax - coordinates.ymin)/M
    dx = (coordinates.xmax - coordinates.xmin)/L
    
    
    for m in range(1, M+1):
        for l in range(1, L+1):
            for k in range(M-m+1):
                for j in range(L-l+1):
                    if m == l:
                        y0 = coordinates.ymin + k*dy
                        y1 = y0 + m*dy

                        x0 = coordinates.xmin + j*dx
                        x1 = x0 + l*dx
                    
                        subwindows.append(Window(
                            Rectangle(int(m), int(l), int(M), int(L)),
                            Coordinates(int(y0), int(y1), int(x0), int(x1))
                        ))
                    
    return subwindows
