from collections import namedtuple

Coordinates = namedtuple('Coordinates', ('ymin', 'ymax', 'xmin', 'xmax'))

MAX_STEPS = 200
GRACE_PERIOD = 40
REWARD_THRESHOLD = 0.3

DOT_SIZE = 100

MAP_CODE = {
    'background' : 0, #black
    'uncounted' : 1, #white
    'enclosed' : 2, #red
    'unenclosed': 3, #blue
}

NET_ARCH = [20, 20]

ACTION_CODE = {
    0: [(1, 1, -1, -1), 'down_left'],
    1: [(1, 1, 0, 0), 'down'],
    2: [(1, 1, 1, 1), 'down_right'],
    3: [(0, 0, 1, 1), 'right'],
    4: [(-1, -1, 1, 1), 'up_right'],
    5: [(-1, -1, 0, 0), 'up'],
    6: [(-1, -1, -1, -1), 'up_left'],
    7: [(0, 0, -1, -1), 'left'],
    8: [(1, 0, 0, 0), 'decrease_ymin'],
    9: [(-1, 0, 0, 0), 'increase_ymin'],
    10: [(0, 1, 0, 0), 'decrease_ymax'],
    11: [(0, -1, 0, 0), 'increase_ymax'],
    12: [(0, 0, 1, 0), 'increase_xmin'],
    13: [(0, 0, -1, 0), 'decrease_xmin'],
    14: [(0, 0, 0, 1), 'increase_xmax'],
    15: [(0, 0, 0, -1), 'decrease_xmax'],
    16: [(0, 0, 0, 0), 'nothing'],
}

REVERSE_ACTION_CODE = {v[1]: [v[0], k] for k, v in ACTION_CODE.items()}

SHIFT_ACTIONS = {n for n in range(8)}
SIZE_CHANGE_ACTION = {n for n in range(8, 16)}
NO_ACTION = {12}

PARAMETERS = {
    'LAMBDA_PIXELS': 1, # for pixel ratio reward
    'LAMBDA_POLYGONS': 1, # for polygon reward
    'LAMBDA_WINDOW': 1, # for window order reward
    'SHIFT_STEPSIZE': 5,
    'SIZE_CHANGE_STEPSIZE': 5
}

WINDOW_CONFIG = {
    'NUM_ROWS': 2,
    'NUM_COLS': 2,
    'LENGTH': 100,
    'HEIGHT': 100,
}

NUM_SUBWINDOWS = 0
for d in range(min(WINDOW_CONFIG['NUM_ROWS'], WINDOW_CONFIG['NUM_COLS'])):
    num_r, num_c = WINDOW_CONFIG['NUM_ROWS'] - d, WINDOW_CONFIG['NUM_COLS'] - d
    NUM_SUBWINDOWS += num_r * num_c
