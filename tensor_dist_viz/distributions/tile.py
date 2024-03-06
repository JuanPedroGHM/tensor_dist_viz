from tensor_dist_viz.distributions.dist import Dist
import numpy as np

from tensor_dist_viz.tensor import Tensor

class CyclicDist(Dist):
    def __init__(self, num_processors: int, block_size: int) -> None:
        self._num_processors = num_processors
        self._block_size = block_size
    
    @property
    def numProcessors(self) -> int:
        return self._num_processors
    
    @property
    def processorArrangement(self) -> np.ndarray:
        return np.array((self._num_processors, 1))
    
    def getProcessorMultiIndex(self, index: int) -> np.ndarray:
        return np.array((index, ))
    
    def getProcessorView(self, tensor: Tensor) -> np.ndarray:
        if tensor.shape[0] % self._block_size != 0:
            raise ValueError("The tensor dimensions must be divisible by the block size")
        
        processor_view = np.zeros((tensor.shape[0], self._num_processors), dtype=np.bool_)
        for i in range(tensor.shape[0]):
            processor_view[i, i % self._num_processors] = True
        
        return processor_view

