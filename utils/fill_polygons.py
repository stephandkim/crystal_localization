from utils import * 
from src import *


# def flood_fill(img_map, y, x, windows):
#     if img_map is None:
#         print('The img map is not ready. \n')
#         return
    
#     if ((y >= img_map.shape[0])
#         or (y < 0)
#         or (x >= img_map.shape[1])
#         or (x < 0)
#         or (img_map[y, x] != constants.MAP_CODE['uncounted'])
#        ):
        
#         print('Selection error.')
#         return

#     enclosed_queue = queue.Queue()
#     unenclosed_queue = queue.Queue()
#     enclosed = True
#     polygon = Polygon(windows)
    
#     enclosed_queue.put([y, x])
#     while ((not enclosed_queue.empty()) or (not unenclosed_queue.empty())):
#         y, x = enclosed_queue.get() if (not enclosed_queue.empty()) else unenclosed_queue.get()
#         if ((y >= img_map.shape[0])
#             or (y < 0)
#             or (x >= img_map.shape[1])
#             or (x < 0)):
#             enclosed = False
#             continue
        
#         if enclosed:
#             if ((img_map[y, x] == constants.MAP_CODE['background']) 
#                 or (img_map[y, x] == constants.MAP_CODE['enclosed'])
#                ):
#                 continue
#             else:
#                 img_map[y, x] = constants.MAP_CODE['enclosed']
                
#                 for idx, window in enumerate(windows):
#                     if ((y>window.coordinates.ymin)
#                         and (y<window.coordinates.ymax)
#                         and (x>window.coordinates.xmin)
#                         and (x<window.coordinates.xmax)
#                     ):
#                         polygon.pixel_location_info[idx] += 1
                        
#                 enclosed_queue.put([y+1, x])
#                 enclosed_queue.put([y-1, x])
#                 enclosed_queue.put([y, x+1])
#                 enclosed_queue.put([y, x-1])
#         else:
#             enclosed_queue.queue.clear()
#             polygon = None
            
#             if ((img_map[y, x] == constants.MAP_CODE['background']) 
#                 or (img_map[y, x] == constants.MAP_CODE['unenclosed'])
#                ):
#                 continue
#             else:
#                 img_map[y, x] = constants.MAP_CODE['unenclosed']
#                 unenclosed_queue.put([y+1, x])
#                 unenclosed_queue.put([y-1, x])
#                 unenclosed_queue.put([y, x+1])
#                 unenclosed_queue.put([y, x-1])
            
            
#     return polygon, enclosed




# def unfill_unenclosed(img_map, y, x):
    
#     if img_map is None:
#         print('The img map is not ready. \n')
#         return
    
#     if ((y >= img_map.shape[0])
#         or (y < 0)
#         or (x >= img_map.shape[1])
#         or (x < 0)
#        ):
        
#         print('Selection error.')
#         return
    
#     queued = queue.Queue()
#     queued.put([y, x])
    
#     while not queued.empty():
#         y, x = queued.get()
    
#         if ((y >= img_map.shape[0])
#             or (y < 0)
#             or (x >= img_map.shape[1])
#             or (x < 0)
#             or (img_map[y, x] == constants.MAP_CODE['background'])
#             or (img_map[y, x] == constants.MAP_CODE['uncounted'])
#             or (img_map[y, x] == constants.MAP_CODE['enclosed'])
#            ):
#             continue
#         else:
#             img_map[y, x] = constants.MAP_CODE['uncounted']
#             queued.put([y+1, x])
#             queued.put([y-1, x])
#             queued.put([y, x+1])
#             queued.put([y, x-1])
        
#     return img_map



# def unfill_enclosed(img_map, y, x):
    
#     if img_map is None:
#         print('The img map is not ready. \n')
#         return
    
#     if ((y >= img_map.shape[0])
#         or (y < 0)
#         or (x >= img_map.shape[1])
#         or (x < 0)
#        ):
        
#         print('Selection error.')
#         return
    
#     queued = queue.Queue()
#     queued.put([y, x])
    
#     while not queued.empty():
#         y, x = queued.get()
    
#         if ((y >= img_map.shape[0])
#             or (y < 0)
#             or (x >= img_map.shape[1])
#             or (x < 0)
#             or (img_map[y, x] == constants.MAP_CODE['background'])
#             or (img_map[y, x] == constants.MAP_CODE['uncounted'])
#             or (img_map[y, x] == constants.MAP_CODE['unenclosed'])
#            ):
#             continue
#         else:
#             img_map[y, x] = constants.MAP_CODE['background']
#             queued.put([y+1, x])
#             queued.put([y-1, x])
#             queued.put([y, x+1])
#             queued.put([y, x-1])
        
#     return img_map
