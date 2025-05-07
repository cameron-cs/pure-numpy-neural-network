from typing import Tuple
import numpy as np


def xavier_init(shape):
    return np.random.randn(*shape) * np.sqrt(2. / (shape[0] + shape[1]))


def he_init(shape):
    return np.random.randn(*shape) * np.sqrt(2. / shape[0])


def uniform_init(shape, low=-0.1, high=0.1):
    return np.random.uniform(low, high, size=shape)


def normal_init(shape, mean=0.0, std=1.0):
    return np.random.normal(loc=mean, scale=std, size=shape)


def unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """
    Broadcasts a gradient to the required shape, ensuring dimensions are handled correctly.

    Args:
        grad (np.ndarray): The gradient to be broadcasted.
        shape (Tuple[int, ...]): The target shape to which the gradient should be broadcasted.

    Returns:
        np.ndarray: The broadcasted gradient.
    """
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad
