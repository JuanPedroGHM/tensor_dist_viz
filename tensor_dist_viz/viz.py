import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensor_dist_viz.tensor import Tensor

import networkx as nx

def plotTensor2D(tensor: Tensor, dims: None | tuple = None) -> None:
    if tensor.order != 2 and not dims and len(dims) != 2:
        raise ValueError("Only 2D tensors are supported, please provide the dimensions to print")

    if dims is None:
        dims = np.array((0, 1))
    else:
        dims = np.array(dims)

    img_shape = tensor.shape[dims]
    img = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.int32) * 100
    img[0, :, :] = 0
    img[1, :, :] = 50
    img[2, :, :] = 100

    img[:, 0, :] += 0
    img[:, 1, :] += 25
    img[:, 2, :] += 50
    img[:, 3, :] += 75
    
    plt.imshow(img[::-1, :, :], origin="lower", aspect="equal")
    axs = plt.gca() 

    axs.set_xticks(np.arange(0, img_shape[1], 1))
    axs.set_yticks(np.arange(0, img_shape[0], 1))
    
    axs.set_yticklabels(np.arange(0, img_shape[0], 1)[::-1])

    axs.set_xticks(np.arange(-.5, img_shape[1], 1), minor=True)
    axs.set_yticks(np.arange(-.5, img_shape[0], 1), minor=True)
    
    axs.grid(which="minor", color="black", linestyle="-", linewidth=1)
    axs.tick_params(which="minor", bottom=False, left=False)
    plt.show()
    

def getNColors(n: int, colormap: str = "viridis") -> np.ndarray:
    return mpl.colormaps[colormap].resampled(n).colors

def rgba2hex(rgba: np.ndarray) -> str:
    RGBA = rgba * 255
    RGBA = RGBA.astype(np.uint8)
    return "#{:02x}{:02x}{:02x}{:02x}".format(*RGBA)

def plot2DMesh(mesh: Tensor) -> None:
    if mesh.order == 1:
        graph = nx.grid_2d_graph(mesh.size, 1)
    elif mesh.order == 2:
        graph = nx.grid_2d_graph(*mesh.shape)
    else:
        raise ValueError("Only 1D and 2D meshes are supported")

    colors = getNColors(mesh.size)
    hexColors = [rgba2hex(color) for color in colors]
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, node_color=hexColors)
    plt.show()