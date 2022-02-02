from src import *
from utils import *


def plot_local_map(env, coordinates, ax=None):
    local_map = env.env_map[
        coordinates.ymin:coordinates.ymax,
        coordinates.xmin:coordinates.xmax
    ].copy()
    
    if ax:
        ax.imshow(local_map)
    else:
        plt.imshow(local_map)

        
def plot_local_maps(env):
    fig, axs = plt.subplots(ncols=3, nrows=5, figsize=(20,30))
    
    for idx, window in enumerate(env.windows):
        plot_local_map(env, window.coordinates, axs.flatten()[idx])

#     fig.delaxes(axs.flatten()[-1])
    axs.flatten()[-1].set_visible(False)
    plt.tight_layout()

