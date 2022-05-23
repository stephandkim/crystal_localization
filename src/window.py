from collections import deque
import src.config as config
import math
import matplotlib.pyplot as plt


class Polygon(object):
    def __init__(self):
        self.type = None # 'enclosed' or 'unenclosed'
        self.pixel_info = None
        self.total_pixels = None

    def reset_pixel_info(self, num_subwindows):
        self.pixel_info = [0] * num_subwindows

class Window(object):
    def __init__(self, image, coordinates=config.Coordinates(0, 10, 0, 10), window_config=config.WINDOW_CONFIG):
        # Coordinates are for the location of the window. Row and col values for creating subwindows.
        self.abs_coordinates = coordinates
        self.num_rows = window_config['NUM_ROWS']
        self.num_cols = window_config['NUM_COLS']

        # This image will be used for counting via the flood fill algorithm.
        self.image_for_counting = image[
            self.abs_coordinates.ymin:self.abs_coordinates.ymax,
            self.abs_coordinates.xmin:self.abs_coordinates.xmax
        ].copy()

        # Subwindows are dict, polygons are lists. A polygon(list) contains pixel information for each subwindow.
        self.subwindows, self.polygons = [], []

    def create_subwindows(self) -> None:
        # Create subwindows and update self.subwindows.
        dy, dx = (self.abs_coordinates.ymax - self.abs_coordinates.ymin) / self.num_rows, \
                 (self.abs_coordinates.xmax - self.abs_coordinates.xmin) / self.num_cols
        order = 0

        for r_end in range(1, self.num_rows+1):
            for c_end in range(1, self.num_cols+1):
                if r_end == c_end:
                    order += 1
                    for r_start in range(self.num_rows-r_end+1):
                        for c_start in range(self.num_cols-c_end+1):

                            # ymin = self.coordinates.ymin + r_start * dy
                            ymin = int(r_start * dy)
                            ymax = int(ymin + r_end * dy)
                            # xmin = self.coordinates.xmin + c_start * dx
                            xmin = int(c_start * dx)
                            xmax = int(xmin + c_end * dx)

                            self.subwindows.append(
                                {
                                    'rel_coordinates': config.Coordinates(ymin, ymax, xmin, xmax),
                                    'abs_coordinates': config.Coordinates(
                                                        self.abs_coordinates.ymin + ymin,
                                                        self.abs_coordinates.ymin + ymax,
                                                        self.abs_coordinates.xmin + xmin,
                                                        self.abs_coordinates.xmin + xmax
                                                        ),
                                    'order': order,
                                    'enclosed': {'num_polygons': 0,
                                                 'pixels_polygons': 0,
                                                 'score_pixels': 0,
                                                 'score_polygons': 0,
                                                 },
                                    'unenclosed': {'num_polygons': 0,
                                                   'pixels_polygons': 0,
                                                   'score_pixels': 0,
                                                   'score_polygons': 0,
                                                   },
                                    'score_window': 0,
                                    'area': 0,
                                }
                            )

    def flood_fill(self, y_local, x_local, num_subwindows):
        # In case of no input image.
        if self.image_for_counting is None:
            return None

        # In case of initial coordinate out of bound.
        rows, cols = len(self.image_for_counting), len(self.image_for_counting[0])
        if (
            y_local < 0 or y_local >= rows or
            x_local < 0 or x_local >= cols or
            self.image_for_counting[y_local][x_local] != config.MAP_CODE['uncounted']
        ):
            return None

        # Two queues for bfs for enclosed and unenclosed polygons.
        enclosed_queue = deque()
        unenclosed_queue = deque()
        enclosed = True
        enclosed_queue.append((y_local, x_local))
        polygon = Polygon()
        polygon.reset_pixel_info(num_subwindows=num_subwindows)
        polygon.type = 'enclosed'

        while enclosed_queue or unenclosed_queue:
            y_local, x_local = enclosed_queue.pop() if enclosed_queue else unenclosed_queue.pop()
            if (
                y_local < 0 or y_local >= rows or
                x_local < 0 or x_local >= cols
            ):
                enclosed = False
                continue

            if enclosed:
                if (
                    self.image_for_counting[y_local][x_local] == config.MAP_CODE['background'] or
                    self.image_for_counting[y_local][x_local] == config.MAP_CODE['enclosed']
                ):
                    continue
                else:
                    self.image_for_counting[y_local][x_local] = config.MAP_CODE['enclosed']

                    for (j, i) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        enclosed_queue.append((y_local + j, x_local + i))
            else:
                enclosed_queue.clear()

                if polygon.type == 'enclosed':
                    polygon.type = 'unenclosed'
                    polygon.reset_pixel_info(num_subwindows=num_subwindows)

                if (
                    self.image_for_counting[y_local][x_local] == config.MAP_CODE['background'] or
                    self.image_for_counting[y_local][x_local] == config.MAP_CODE['unenclosed']
                ):
                    continue
                else:
                    self.image_for_counting[y_local][x_local] = config.MAP_CODE['unenclosed']

                    for (j, i) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        unenclosed_queue.append((y_local + j, x_local + i))

            for idx, subwindow in enumerate(self.subwindows):
                if (
                    subwindow['rel_coordinates'].ymin < y_local < subwindow['rel_coordinates'].ymax and
                    subwindow['rel_coordinates'].xmin < x_local < subwindow['rel_coordinates'].xmax
                ):
                    polygon.pixel_info[idx] += 1

        polygon.total_pixels = max(polygon.pixel_info)
        if polygon.total_pixels <= config.DOT_SIZE:
            return None
        else:
            return polygon

    def count_polygons(self):
        for y in range(len(self.image_for_counting)):
            for x in range(len(self.image_for_counting[0])):
                polygon = self.flood_fill(y, x, num_subwindows=len(self.subwindows))
                if polygon is not None:
                    self.polygons.append(polygon)

        for polygon in self.polygons:
            for idx, val in enumerate(polygon.pixel_info):
                self.subwindows[idx][polygon.type]['pixels_polygons'] += val
                if val == polygon.total_pixels:
                    self.subwindows[idx][polygon.type]['num_polygons'] += 1

    def calculate_scores(self):
        window_max_order = self.subwindows[-1]['order']
        for subwindow in self.subwindows:
            # Score from pixel ratio
            subwindow['area'] = abs(subwindow['rel_coordinates'].ymin - subwindow['rel_coordinates'].ymax) * \
                abs(subwindow['rel_coordinates'].xmin - subwindow['rel_coordinates'].xmax)

            for polygon_type in ['enclosed', 'unenclosed']:
                subwindow[polygon_type]['pixel_ratio'] = subwindow[polygon_type]['pixels_polygons'] / subwindow['area']

                subwindow[polygon_type]['score_pixels'] = subwindow[polygon_type]['pixel_ratio']

                # Score from the number of polygons
                subwindow[polygon_type]['score_polygons'] = 1 / subwindow[polygon_type]['num_polygons'] if subwindow[polygon_type]['num_polygons'] >= 1 \
                 else 0

            # Score from window size
            subwindow['score_window'] = math.exp(
                -1 * config.PARAMETERS['LAMBDA_WINDOW'] * (1 - subwindow['order'] / window_max_order)
            )

    def draw_lines(self, image, idx_list: list[int]):
        for idx in idx_list:
            subwindow = self.subwindows[idx]
            coordinates = subwindow['abs_coordinates']

            val = 1
            for y_local in range(coordinates.ymin, coordinates.ymax):
                image[y_local][coordinates.xmin] = val
                image[y_local][coordinates.xmax-1] = val
            for x_local in range(coordinates.xmin, coordinates.xmax):
                image[coordinates.ymin][x_local] = val
                image[coordinates.ymax-1][x_local] = val

        return image
