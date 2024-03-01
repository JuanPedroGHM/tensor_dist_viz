import numpy as np

_order2npOrder = {
    "C": "F",
    "R": "C"
}

def multi2linearIndex(dims: np.ndarray, indices: np.ndarray, order: np.ndarray = None) -> int:
    if len(indices) != len(dims):
        raise ValueError("Indices must have the same length as the tensor's dimensions")
    
    result: int = 0
    if order is None:
        for i in range(len(indices)):
            result += indices[i] * np.prod(dims[:i])
    else:
        indices = indices[order]
        dims = dims[order]
        for i in range(len(indices)):
            result += indices[i] * np.prod(dims[:i])
    return result

def order2npOrder(order: str) -> str:
    return _order2npOrder[order]
