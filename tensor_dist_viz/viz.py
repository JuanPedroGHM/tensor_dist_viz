import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensor_dist_viz.distributions.dist import Dist
from tensor_dist_viz.tensor import Tensor
from tensor_dist_viz.distributions.pmesh import PMeshDist

import networkx as nx

def drawColorBar(fig, axs, colors: list[np.ndarray]):
    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.BoundaryNorm(np.arange(-0.5, len(colors), 1), cmap.N)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axs, orientation="horizontal", ticks=np.arange(0, len(colors), 1), location="bottom")
    cbar.ax.set_xticklabels(np.arange(0, len(colors), 1))
    cbar.set_label("Processor index")

def plotProcessorView2D(tensor: Tensor, distribution: Dist, cbar: bool = True) -> None:
    if tensor.order >2:
        raise ValueError("Only 2D tensors are supported, please provide the dimensions to print")
    
    if len(distribution.processorArrangement) > 2:
        raise ValueError("Only 2D meshes are supported")
    
    processor_view = distribution.processorView(tensor)
    
    if tensor.order == 1:
        img_shape = tensor.shape.reshape(-1, 1)
    else:
        img_shape = tensor.shape


    processorArragement = distribution.processorArrangement
    subplot_x = processorArragement[0]
    subplot_y = processorArragement[1] if len(distribution.processorArrangement) > 1 else 1
    
    colors = getNColors(distribution.numProcessors)
    fig, axs = plt.subplots(nrows=subplot_x, ncols=subplot_y, figsize=(subplot_x * 2.5, subplot_y * 2.5), sharex=True, sharey=True)
    
    for p in range(distribution.numProcessors):
        p_midx = distribution.getProcessorMultiIndex(p)

        img = np.apply_along_axis(lambda a: colors[p] if a[p] else [0.0, 0.0, 0.0, 0.0], -1, processor_view)
        axs[p_midx[0], p_midx[1]].imshow(img, origin="upper", aspect="equal")

        # Ticks
        axs[p_midx[0], p_midx[1]].set_xticks(np.arange(0.0, img_shape[1], 1.0))
        axs[p_midx[0], p_midx[1]].set_yticks(np.arange(0.0, img_shape[0], 1.0))

        axs[p_midx[0], p_midx[1]].set_xticks(np.arange(-.5, img_shape[1], 1.0), minor=True)
        axs[p_midx[0], p_midx[1]].set_yticks(np.arange(-.5, img_shape[0], 1.0), minor=True)
        
        axs[p_midx[0], p_midx[1]].grid(which="minor", color="black", linestyle="-", linewidth=1)
        axs[p_midx[0], p_midx[1]].tick_params(which="minor", bottom=False, left=False)
        axs[p_midx[0], p_midx[1]].tick_params(which="major", bottom=False, left=False)

        # Labels
        axs[p_midx[0], p_midx[1]].set_xticklabels([])
        axs[p_midx[0], p_midx[1]].set_yticklabels([])
        
        axs[p_midx[0], p_midx[1]].set_xlabel("Axis 1")
        axs[p_midx[0], p_midx[1]].set_ylabel("Axis 0")
        axs[p_midx[0], p_midx[1]].title.set_text(f"P {p_midx}")
        
    ## Add discrete colorbar with the processor index and colors

    if cbar:
        drawColorBar(fig, axs, colors)


    plt.show()

def plotTensor2D(tensor: Tensor, distribution: PMeshDist, cbar: bool = True) -> None:
    if tensor.order >2:
        raise ValueError("Only 2D tensors are supported, please provide the dimensions to print")
    
    if len(distribution.processorArrangement) > 2:
        raise ValueError("Only 2D meshes are supported")
    
    processor_view = distribution.processorView(tensor)
    
    if tensor.order == 1:
        img_shape = tensor.shape.reshape(-1, 1)
    else:
        img_shape = tensor.shape
    
    colors = getNColors(distribution.numProcessors)
    img = np.zeros((*img_shape, 4))
    for i in range(tensor.size):
        m_idx = tensor.getMultiIndex(i)
        img[m_idx[0], m_idx[1], :] = colors[np.argmax(processor_view[m_idx[0], m_idx[1], :])]
        
    plt.figure(figsize=(5, 5))
    axis = plt.gca()
    axis.imshow(img[::-1], origin="lower", aspect="equal")

    # Ticks
    axis.set_xticks(np.arange(0, img_shape[1], 1))
    axis.set_yticks(np.arange(0, img_shape[0], 1))
    
    axis.set_xticks(np.arange(-.5, img_shape[1], 1), minor=True)
    axis.set_yticks(np.arange(-.5, img_shape[0], 1), minor=True)
    
    axis.grid(which="minor", color="black", linestyle="-", linewidth=1)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.tick_params(which="major", bottom=False, left=False)

    # Labels
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    
    axis.set_xlabel("Axis 1")
    axis.set_ylabel("Axis 0")

    if cbar:
        drawColorBar(plt.gcf(), axis, colors)

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


def explode(data: np.ndarray) -> np.ndarray:
    size = np.array(data.shape)*2
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e
    
def plotTensor3D(tensor: Tensor, distribution: Dist, cbar: bool = True) -> None:

    if tensor.order != 3:
        raise ValueError("Only 3D tensors are supported")
    
    processorView = distribution.processorView(tensor) 
    colors = getNColors(distribution.numProcessors)
    colors_hex = [rgba2hex(color) for color in colors]
    colors_edges = [rgba2hex(color * 0.8) for color in colors]
    
    x, y, z = np.indices(np.array(tensor.shape))

    # build up the numpy logo
    filled = np.ones(tensor.shape)
    facecolors = np.apply_along_axis(lambda a: colors_hex[np.argmax(a)], -1, processorView)
    edgecolors = np.apply_along_axis(lambda a: colors_edges[np.argmax(a)], -1, processorView)

    # upscale the above voxel image, leaving gaps
    filled_2 = explode(filled)
    fcolors_2 = explode(facecolors)
    ecolors_2 = explode(edgecolors)

    # Shrink the gaps
    x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
    x[0::2, :, :] += 0.1
    y[:, 0::2, :] += 0.1
    z[:, :, 0::2] += 0.1
    x[1::2, :, :] += 0.9
    y[:, 1::2, :] += 0.9
    z[:, :, 1::2] += 0.9

    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(x, y, z, filled_2, facecolors=fcolors_2, edgecolors=ecolors_2)
    ax.view_init(25, -135, 0)
    ax.set_proj_type('persp')
    ax.set_xlabel('Axis 0')
    ax.set_ylabel('Axis 1')
    ax.set_zlabel('Axis 2')
    ax.set_aspect('equal')
    ax.grid(False)

    if cbar:
        drawColorBar(plt.gcf(), ax, colors)
    
    plt.show()