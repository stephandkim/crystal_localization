from src import *
from utils import *
import random
import torch
import os
import json
import math
from collections import deque

class Environment(object):
    
    def __init__(self, h, l, env_map, square):
        # Environment information
        self.env_map = env_map.copy()
        self.local_map = None # For counting polygons
        self.square = square
        self.limits = Coordinates(0, self.env_map.shape[0], 0, self.env_map.shape[1])
        self.h = h
        self.l = l
             
        # Termination conditions
        self.termination = False
        self.turn_max = 100
        self.action_tracker = deque(maxlen=10)
        self.total_reward_tracker = deque(maxlen=10)
        
        # For starting locations
        uncounted_pixels = np.where(self.env_map == constants.MAP_CODE['uncounted'])
        self.starting_loc_candidates = list(zip(uncounted_pixels[0], uncounted_pixels[1]))
        
        # Initiallizing variables
        self.state = torch.zeros((sum(m ** 2 for m in range(1, self.square.M+1)))*2, dtype=torch.float).view(1,-1)
        self.coordinates = None
        self.polygons = []
        self.windows = []
        self.total_reward = 0
        
    def reset(self):
        # Clearing termination conditions
        self.action_tracker.clear()
        self.total_reward_tracker.clear()
        
        # Getting new starting location
        allowed = False
        while not allowed:
            y, x = self.starting_loc_candidates[
                random.randint(0, len(self.starting_loc_candidates)-1)
            ]
            if ((y+self.h>self.limits.ymax)
                or (x+self.l>self.limits.xmax)
               ):
                allowed = False
            else:
                allowed = True
        
        # Getting new coordinates and the corresponding windows
        self.coordinates = Coordinates(y, y+self.h, x, x+self.l)
        self.refresh_windows()
        self.refresh_local_map()
        
    def refresh_windows(self):
        self.polygons = []
        self.windows = []
        self.windows, self.windows_hierarchy = get_windows(self.square.M, self.coordinates)
        
        
    def refresh_local_map(self):
        # Prepare a local map for flood fill counting
        self.local_map = self.env_map[
            self.coordinates.ymin:self.coordinates.ymax,
            self.coordinates.xmin:self.coordinates.xmax
        ].copy()
        
    def check_termination(self, turn):
        # For loops
        if (turn >= self.turn_max):
            self.termination = True
            print('TERMINATION: max turn')
            
        if (
            (len(self.action_tracker)==10) 
            and (len([idx for idx, val in enumerate(self.action_tracker) if val==constants.ACTION_CODE['nothing']]) > 7)
           ):
            self.termination = True
            print('TERMINATION: doing nothing')
        

    def count_polygons(self):
        for y in range(self.local_map.shape[0]):
            for x in range(self.local_map.shape[1]):
                if self.local_map[y, x] == constants.MAP_CODE['uncounted']:
                    self.local_map, polygon, enclosed = flood_fill(self.local_map, y, x, self.windows)
                    if enclosed:
                        self.polygons.append(polygon)
                        
        
    def update_windows(self):
        if len(self.polygons) == 0:
            print('No polygons.')
            return
        
        for polygon in self.polygons:
            for idx, pix in enumerate(polygon.pixel_location_info):
                if pix == polygon.pixel_location_info[-1]:
                    self.windows[idx].polygons_enclosed += 1
                    total_pix = (self.windows[idx].coordinates.ymax-self.windows[idx].coordinates.ymin) * \
                                (self.windows[idx].coordinates.xmax-self.windows[idx].coordinates.xmin)
                    self.windows[idx].pixel_ratio += pix/total_pix

                    
    def step(self, action):
        # Updating the environment according to the action of the agent
        self.move(action)
        self.refresh_windows()
        self.refresh_local_map()
        self.count_polygons()
        self.update_windows()
        
        state = np.zeros((len(self.windows), 2))
        self.total_reward = 0
        
        # Calculating the reward for each window
        for idx, window in enumerate(self.windows):
            window.r_pixel = 0 if window.pixel_ratio == 0 else math.exp(-1 * constants.LAMBDA_X * (1-window.pixel_ratio))
            window.r_polygons = 0 if window.polygons_enclosed == 0 else \
                        math.exp(-1 * constants.LAMBDA_P * (window.polygons_enclosed - 1))
            window.r_square = math.exp(-1 * constants.LAMBDA_S * (1-window.square.m/window.square.M))
            window.r_total = window.r_pixel * window.r_polygons * window.r_square
            
            self.total_reward += window.r_total
            state[idx][0] = window.r_total
            state[idx][1] = window.pixel_ratio
        self.total_reward = self.total_reward / len(self.windows)
        
        return torch.tensor(state, dtype=torch.float).view(1,-1), self.total_reward
    
    def save_replay(self, path, episode, action):
        # Saving coordinates for replay for replay and debugging
        info = {
            'square' : self.windows[-1].square,
            'coordinates' : self.windows[-1].coordinates,
            'action': action,
            'r_total': self.windows[-1].r_total,
            'r_pixel': self.windows[-1].r_pixel,
            'r_polygons': self.windows[-1].r_polygons,
            'r_square': self.windows[-1].r_square,
            'pixel_ratio': self.windows[-1].pixel_ratio,
            'polygons_enclosed': self.windows[-1].polygons_enclosed,
        }
        replay_save_path = path+'/replay'
        if not os.path.exists(replay_save_path):
            os.mkdir(replay_save_path)
        with open(replay_save_path+'/episode'+str(episode)+'.json', 'a') as outfile:
            json.dump(info, outfile)
            outfile.write('\n')
            outfile.close()    
        
    def move(self, action):
        # Move the coordinates according to the action
        if action == constants.ACTION_CODE['down_left']:
            new_coordinates = Coordinates(self.coordinates.ymin + constants.SHIFT_STEPSIZE,
                                          self.coordinates.ymax + constants.SHIFT_STEPSIZE,
                                          self.coordinates.xmin - constants.SHIFT_STEPSIZE,
                                          self.coordinates.xmax - constants.SHIFT_STEPSIZE
                                         )
            if ((self.limits.ymax < new_coordinates.ymax)
                or (self.limits.xmin > new_coordinates.xmin)):
                pass
            else:
                self.coordinates = new_coordinates
                
        elif action == constants.ACTION_CODE['down']:
            new_coordinates = Coordinates(self.coordinates.ymin + constants.SHIFT_STEPSIZE,
                                          self.coordinates.ymax + constants.SHIFT_STEPSIZE,
                                          self.coordinates.xmin,
                                          self.coordinates.xmax
                                         )
            if (self.limits.ymax < new_coordinates.ymax):
                pass
            else:
                self.coordinates = new_coordinates
                
        elif action == constants.ACTION_CODE['down_right']:
            new_coordinates = Coordinates(self.coordinates.ymin + constants.SHIFT_STEPSIZE,
                                          self.coordinates.ymax + constants.SHIFT_STEPSIZE,
                                          self.coordinates.xmin + constants.SHIFT_STEPSIZE,
                                          self.coordinates.xmax + constants.SHIFT_STEPSIZE
                                         )
            if ((self.limits.ymax < new_coordinates.ymax)
                or (self.limits.xmax < new_coordinates.xmax)):
                pass
            else:
                self.coordinates = new_coordinates
                
        elif action == constants.ACTION_CODE['right']:
            new_coordinates = Coordinates(self.coordinates.ymin,
                                          self.coordinates.ymax,
                                          self.coordinates.xmin + constants.SHIFT_STEPSIZE,
                                          self.coordinates.xmax + constants.SHIFT_STEPSIZE
                                         )
            if (self.limits.xmax < new_coordinates.xmax):
                pass
            else:
                self.coordinates = new_coordinates
        
        elif action == constants.ACTION_CODE['up_right']:
            new_coordinates = Coordinates(self.coordinates.ymin - constants.SHIFT_STEPSIZE,
                                          self.coordinates.ymax - constants.SHIFT_STEPSIZE,
                                          self.coordinates.xmin + constants.SHIFT_STEPSIZE,
                                          self.coordinates.xmax + constants.SHIFT_STEPSIZE
                                         )
            if ((self.limits.ymin > new_coordinates.ymin)
                or (self.limits.xmax < new_coordinates.xmax)):
                pass
            else:
                self.coordinates = new_coordinates
                
                
        elif action == constants.ACTION_CODE['up']:
            new_coordinates = Coordinates(self.coordinates.ymin - constants.SHIFT_STEPSIZE,
                                          self.coordinates.ymax - constants.SHIFT_STEPSIZE,
                                          self.coordinates.xmin,
                                          self.coordinates.xmax
                                         )
            if (self.limits.ymin > new_coordinates.ymin):
                pass
            else:
                self.coordinates = new_coordinates
                
                
        elif action == constants.ACTION_CODE['up_left']:
            new_coordinates = Coordinates(self.coordinates.ymin - constants.SHIFT_STEPSIZE,
                                          self.coordinates.ymax - constants.SHIFT_STEPSIZE,
                                          self.coordinates.xmin - constants.SHIFT_STEPSIZE,
                                          self.coordinates.xmax - constants.SHIFT_STEPSIZE
                                         )
            if ((self.limits.ymin > new_coordinates.ymin)
                or (self.limits.xmin > new_coordinates.xmin)):
                pass
            else:
                self.coordinates = new_coordinates
                
                
        elif action == constants.ACTION_CODE['left']:
            new_coordinates = Coordinates(self.coordinates.ymin,
                                          self.coordinates.ymax,
                                          self.coordinates.xmin - constants.SHIFT_STEPSIZE,
                                          self.coordinates.xmax - constants.SHIFT_STEPSIZE
                                         )
            if (self.limits.xmin > new_coordinates.xmin):
                pass
            else:
                self.coordinates = new_coordinates
                
        
        elif action == constants.ACTION_CODE['zoom_in']:
            new_coordinates = Coordinates(self.coordinates.ymin + constants.ZOOM_STEPSIZE,
                                          self.coordinates.ymax - constants.ZOOM_STEPSIZE,
                                          self.coordinates.xmin + constants.ZOOM_STEPSIZE,
                                          self.coordinates.xmax - constants.ZOOM_STEPSIZE
                                         )
            if (int(abs(new_coordinates.ymax-new_coordinates.ymin)/self.square.M)<=2
                or int(abs(new_coordinates.xmax-new_coordinates.xmin)/self.square.M)<=2
               ):
                pass
            else:
                self.coordinates = new_coordinates
                
                
        elif action == constants.ACTION_CODE['zoom_out']:
            ymin_new = self.coordinates.ymin - constants.SHIFT_STEPSIZE
            ymax_new = self.coordinates.ymax + constants.SHIFT_STEPSIZE
            xmin_new = self.coordinates.xmin - constants.SHIFT_STEPSIZE
            xmax_new = self.coordinates.xmax + constants.SHIFT_STEPSIZE
            
            if (self.limits.ymin > ymin_new):
                ymin_new = self.limits.ymin
            if (self.limits.ymax < ymax_new):
                ymax_new = self.limits.ymax
            if (self.limits.xmin > xmin_new):
                xmin_new = self.limits.xmin
            if (self.limits.xmax < xmax_new):
                xmax_new = self.limits.xmax
                
            self.coordinates = Coordinates(ymin_new, ymax_new, xmin_new, xmax_new)
  
