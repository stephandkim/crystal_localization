import gym
from gym import spaces
from gym.utils import seeding

import src.config as config
import src.window as window
import numpy as np


class XtalEnv(gym.Env):
    def __init__(self, images, save_replay=False):
        super(XtalEnv, self).__init__()
        self.images = images # a list of npy images
        self.image = None
        self.image_id = None
        self.window = None
        self.limits = None

        self.rng = None
        self.seed_val = None
        self.seed()
        self.start_loc = None

        self.scores = None
        self.borders = None
        self.state = None
        self.steps = None
        self.cumm_reward = None

        self.replay = None
        self.save_replay = save_replay # To save replays for environments under evaluation

        self.observation_space = spaces.Box(low=0, high=1, shape=(config.NUM_SUBWINDOWS*4 + 4,))
        self.action_space = spaces.Discrete(len(config.ACTION_CODE))
        self.reward_range = (-1, 1)

    def seed(self, seed_val=None):
        self.seed_val = seed_val
        self.rng, seed = seeding.np_random(self.seed_val)
        return [seed]

    def reset(self, start_full_image=False):
        if not self.rng:
            print('self.seed not run yet.')
            return

        self.image_id = self.rng.randint(0, len(self.images))
        self.image = self.images[self.image_id]
        self.limits = config.Coordinates(0, len(self.image), 0, len(self.image[0]))

        if self.window:
            del self.window

        self.steps = 0
        self.cumm_reward = 0

        if self.replay:
            del self.replay
        self.replay = []

        if start_full_image:
            start_coordinates = config.Coordinates(self.limits.ymin,
                                                   self.limits.ymax,
                                                   self.limits.xmin,
                                                   self.limits.xmax
                                                   )
        else:
            self.start_loc = (self.rng.randint(self.limits.ymin, self.limits.ymax - config.WINDOW_CONFIG['HEIGHT']),
                              self.rng.randint(self.limits.xmin, self.limits.xmax - config.WINDOW_CONFIG['LENGTH']),
                              )
            start_coordinates = config.Coordinates(self.start_loc[0],
                                                   self.start_loc[0] + config.WINDOW_CONFIG['HEIGHT'],
                                                   self.start_loc[1],
                                                   self.start_loc[1] + config.WINDOW_CONFIG['LENGTH']
                                                   )

        self.window = window.Window(image=self.image, coordinates=start_coordinates)
        self.window.create_subwindows()
        self.window.count_polygons()
        self.window.calculate_scores()

        self.scores = np.zeros(config.NUM_SUBWINDOWS*4, dtype=np.float32)
        for idx1, subwindow in enumerate(self.window.subwindows):
            criteria = [subwindow['enclosed']['score_polygons'],
                        subwindow['enclosed']['score_pixels'],
                        subwindow['unenclosed']['score_polygons'],
                        subwindow['unenclosed']['score_pixels']
                        ]
            for idx2, score in enumerate(criteria):
                self.scores[idx1*len(criteria) + idx2] = score
        self.borders = np.zeros(4, dtype=np.float32)
        self.check_borders()
        self.state = np.concatenate((self.scores, self.borders))

        return np.array(self.state)

    def step(self, action):
        hit_borders, new_coordinates = self.move(action)
        if self.window:
            del self.window
        self.window = window.Window(image=self.image, coordinates=new_coordinates)
        self.window.create_subwindows()
        self.window.count_polygons()
        self.window.calculate_scores()

        for idx1, subwindow in enumerate(self.window.subwindows):
            criteria = [subwindow['enclosed']['score_polygons'],
                        subwindow['enclosed']['score_pixels'],
                        subwindow['unenclosed']['score_polygons'],
                        subwindow['unenclosed']['score_pixels']
                        ]
            for idx2, score in enumerate(criteria):
                self.scores[idx1*len(criteria) + idx2] = score
        self.check_borders()
        self.state = np.concatenate((self.scores, self.borders))

        if self.window.subwindows[-1]['enclosed']['num_polygons'] < 1:
            reward = -0.1
        elif self.window.subwindows[-1]['enclosed']['num_polygons'] > 1:
            reward = (0.5 / self.window.subwindows[-1]['enclosed']['num_polygons'])
        else:
            reward = (0.5 / self.window.subwindows[-1]['enclosed']['num_polygons']) +\
                (0.5 / (max(1 / self.window.subwindows[-1]['enclosed']['pixel_ratio'], 1/0.7) + (1 - 1/0.7)))


        self.steps += 1
        self.cumm_reward += reward
        if not hit_borders:
            if self.steps >= config.MAX_STEPS:
                done = True
            else:
                done = False
        else:
            done = True
            reward = -1

        if self.save_replay:
            self.replay.append([*self.window.abs_coordinates])

        return self.state, reward, done, {}

    def render(self):
        return

    def close(self):
        if self.window:
            del self.window

    def check_borders(self):
        signs = [-1, 1, -1, 1]
        add_on = [1, 0, 1, 0]

        for idx, limit in enumerate(self.limits):
            if config.PARAMETERS['SHIFT_STEPSIZE'] >= \
             -1 * signs[idx] * self.window.abs_coordinates[idx] + signs[idx] * limit + add_on[idx]:
                self.borders[idx] = 1
            else:
                self.borders[idx] = 0

    def move(self, action):
        new_coordinates = [0, 0, 0, 0]
        for idx, coordinate in enumerate(self.window.abs_coordinates):
            if action in config.SHIFT_ACTIONS or action in config.NO_ACTION:
                new_coordinates[idx] = coordinate + config.ACTION_CODE[action][0][idx] * config.PARAMETERS['SHIFT_STEPSIZE']
            else:
                new_coordinates[idx] = coordinate + config.ACTION_CODE[action][0][idx] * config.PARAMETERS['SIZE_CHANGE_STEPSIZE']

        if (new_coordinates[0] >= self.limits.ymin and
            new_coordinates[1] <= self.limits.ymax and
            new_coordinates[2] >= self.limits.xmin and
            new_coordinates[3] <= self.limits.xmax and
            new_coordinates[0] < new_coordinates[1] and
            new_coordinates[2] < new_coordinates[3]
            ):
            return False, config.Coordinates(new_coordinates[0], new_coordinates[1], new_coordinates[2], new_coordinates[3])
        else:
            return True, self.window.abs_coordinates

