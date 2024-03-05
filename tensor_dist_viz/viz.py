import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensor_dist_viz.tensor import Tensor
from tensor_dist_viz.dist import Distribution

import networkx as nx

def plotProcessorView2D(tensor: Tensor, distribution: Distribution) -> None:
    if tensor.order >2:
        raise ValueError("Only 2D tensors are supported, please provide the dimensions to print")
    
    if distribution._mesh.order > 2:
        raise ValueError("Only 2D meshes are supported")
    
    processor_view = distribution.processorView(tensor)
    
    if tensor.order == 1:
        img_shape = tensor.shape.reshape(-1, 1)
    else:
        img_shape = tensor.shape

    
    subplot_x = distribution._mesh.shape[0]
    subplot_y = distribution._mesh.shape[1] if len(distribution._mesh.shape) > 1 else 1
    

    fig, axs = plt.subplots(nrows=subplot_y, ncols=subplot_x, figsize=(subplot_x * 3, subplot_y * 3))
    
    for p in range(distribution._mesh.size):
        p_midx = distribution._mesh.getMultiIndex(p)
        img = processor_view[:, :, p]
        axs[p_midx[1], p_midx[0]].imshow(img[::-1], origin="lower", aspect="equal", cmap="Greens")
        

        axs[p_midx[1], p_midx[0]].set_xticks(np.arange(0, img_shape[1], 1))
        axs[p_midx[1], p_midx[0]].set_yticks(np.arange(0, img_shape[0], 1))
        
        axs[p_midx[1], p_midx[0]].set_yticklabels(np.arange(0, img_shape[0], 1)[::-1])

        axs[p_midx[1], p_midx[0]].set_xticks(np.arange(-.5, img_shape[1], 1), minor=True)
        axs[p_midx[1], p_midx[0]].set_yticks(np.arange(-.5, img_shape[0], 1), minor=True)
        
        axs[p_midx[1], p_midx[0]].grid(which="minor", color="black", linestyle="-", linewidth=1)
        axs[p_midx[1], p_midx[0]].tick_params(which="minor", bottom=False, left=False)
        axs[p_midx[1], p_midx[0]].title.set_text(f"Processor {p_midx}")
    plt.show()

def plotTensor2D(tensor: Tensor, distribution: Distribution) -> None:
    if tensor.order >2:
        raise ValueError("Only 2D tensors are supported, please provide the dimensions to print")
    
    if distribution._mesh.order > 2:
        raise ValueError("Only 2D meshes are supported")
    
    processor_view = distribution.processorView(tensor)
    
    if tensor.order == 1:
        img_shape = tensor.shape.reshape(-1, 1)
    else:
        img_shape = tensor.shape
    
    colors = getNColors(distribution._mesh.size)
    img = np.zeros((*img_shape, 4))
    for i in range(tensor.size):
        m_idx = tensor.getMultiIndex(i)
        img[m_idx[0], m_idx[1], :] = colors[np.argmax(processor_view[m_idx[0], m_idx[1], :])]
        
    plt.figure(figsize=(5, 5))
    axis = plt.gca()
    axis.imshow(img[::-1], origin="lower", aspect="equal")
    axis = plt.gca()
    axis.set_xticks(np.arange(0, img_shape[1], 1))
    axis.set_yticks(np.arange(0, img_shape[0], 1))
    
    axis.set_yticklabels(np.arange(0, img_shape[0], 1)[::-1])

    axis.set_xticks(np.arange(-.5, img_shape[1], 1), minor=True)
    axis.set_yticks(np.arange(-.5, img_shape[0], 1), minor=True)
    
    axis.grid(which="minor", color="black", linestyle="-", linewidth=1)
    axis.tick_params(which="minor", bottom=False, left=False)
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
    
def plotTensor3D(tensor: Tensor, distribution: Distribution):
    pass