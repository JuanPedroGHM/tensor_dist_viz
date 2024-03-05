from tensor_dist_viz.tensor import Tensor
from tensor_dist_viz.util import multi2linearIndex
import numpy as np

class Distribution:

    def __init__(self, mesh: Tensor, dims_mapping: tuple[tuple], block_sizes: tuple) -> None:
        if len(dims_mapping) != len(block_sizes):
            raise ValueError("The number of dimensions and block sizes must match")
        for dim in dims_mapping:
            for mesh_axis in dim:
                if mesh_axis >= mesh.order or mesh_axis < 0:
                    raise ValueError("The dimension mapping is out of bounds")
    
        for block in block_sizes:
            if block <= 0:
                raise ValueError("The block size must be greater than 0")

        self._mesh = mesh
        self._dims_mapping = dims_mapping
        self._block_sizes = np.array(block_sizes)
        self._omit_replication = None
    
    @property
    def dims_mapping(self) -> tuple[tuple]:
        return self._dims_mapping
    
    @property
    def block_sizes(self) -> tuple:
        return self._block_sizes
    
    def processorView(self, tensor: Tensor) -> np.ndarray:
        if tensor.order != len(self._dims_mapping):
            raise ValueError("The tensor order and the number of dimensions in the distribution must match")
        if not all([tensor.shape[dim] % block == 0 for dim, (mapping, block) in enumerate(zip(self._dims_mapping, self._block_sizes)) if len(mapping) > 0]):
            raise ValueError("The tensor dimensions must be divisible by the block sizes")
        
        
        processor_view = np.ones((*tensor.shape, self._mesh.size), dtype=np.bool_)
        
        dist_axis_list = [self.distributeAxis(i, tensor.shape[i]) for i in range(tensor.order)]
        for i in range(tensor.size):
            m_idx = tensor.getMultiIndex(i)
            for j in range(tensor.order):
                processor_view[*m_idx, :] &= dist_axis_list[j][m_idx[j], :]
            
        return processor_view
    
    def distributeAxis(self, dim: int, dim_size: int) -> np.ndarray:

        mesh_dims_idx = self._dims_mapping[dim]
        num_process = self._mesh.size

        if len(mesh_dims_idx) == 0:
            return np.ones((dim_size, num_process), dtype=np.bool_)

        mesh_dims_idx = Tensor(mesh_dims_idx)
        mesh_dims = Tensor(self._mesh.shape[mesh_dims_idx.shape])
        block_size = self._block_sizes[dim]

        if block_size <= 0:
            raise ValueError("Block size must be greater than 0")

        if block_size > np.floor(dim_size / mesh_dims.size):
            print(mesh_dims.shape)
            print(dim_size)
            print(mesh_dims.size)
            raise ValueError(f"Maximum block size exceeded {block_size} > {np.floor(dim_size / mesh_dims.size)}")
        
        axis_distribution = np.ones((dim_size, num_process), dtype=np.bool_)
        for i in range(dim_size):
            for j in range(num_process):
                p_mi = np.array(self._mesh.getMultiIndex(j))
                t_mi = multi2linearIndex(self._mesh.shape, p_mi, order=mesh_dims_idx.shape)
                u = np.prod(self._mesh.shape[mesh_dims_idx.shape])
                # print(f"{p_mi} : {i} / {block_size}  == {t_mi} (% {u}) -> {np.floor(i / block_size) % u} == {t_mi % u}")
                belongs = np.floor(i / block_size) % u == (t_mi % u)
                #belongs = i % u == (t_mi % u)
                axis_distribution[i, j] = belongs

        return axis_distribution
        

