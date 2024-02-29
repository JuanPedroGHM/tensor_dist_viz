import numpy as np
from tensor_dist_viz.util import multi2linearIndex


class Tensor:

    def __init__(self, dims: tuple | int) -> None:
        if isinstance(dims, int):
            self._dims = np.array((dims,))
        elif len(dims) < 1:
            raise ValueError("Tensor must have at least one dimension")
        elif len(dims) == 1:
            self._dims = np.array((dims[0],))
        else:
            self._dims = np.array(dims)

    @property
    def order(self) -> int:
        return len(self._dims)
    
    @property
    def shape(self) -> np.ndarray:
        return self._dims
    
    @property
    def size(self) -> int:
        return np.prod(self._dims)
    
    def linearIndex(self, indices: tuple, order: str | tuple = "C") -> int:
        if len(indices) != len(self._dims):
            raise ValueError("Indices must have the same length as the tensor's dimensions")
        indices = np.array(indices)

        if order == "C":
            return multi2linearIndex(self._dims, indices)
        elif order == "R":
            return multi2linearIndex(self._dims, indices, order=np.arange(len(self._dims)).transpose())
        elif isinstance(order, (list, tuple)):
            order = np.array(order)
            return multi2linearIndex(self._dims, indices, order)
    
    def info(self) -> None:
        print(f"Order: {self.order}")
        print(f"Shape: {self.shape}")
        print(f"Size: {self.size}")