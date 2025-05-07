from typing import Union, Tuple, List
import numpy as np

from src.nn_utils import unbroadcast


class Tensor:
    """
    A class that represents a Tensor with automatic differentiation features.
    The class supports basic tensor operations like addition, subtraction, multiplication, etc.

    Attributes:
        data (np.ndarray): The data stored in the tensor.
        requires_grad (bool): Whether or not the tensor requires gradient computation.
        grad (np.ndarray or None): The gradient of the tensor (None if no gradients are required).
        _backward (callable): The function that defines the backward pass for this tensor.
        _prev (set): The set of tensors that contributed to this tensor.
        _op (str): The operation that created this tensor.
    """

    def __init__(self, data: Union[np.ndarray, float, int], requires_grad: bool = False,
                 _children: Tuple['Tensor', ...] = (), _op: str = ''):
        """
        Initialize a Tensor object.

        Args:
            data (Union[np.ndarray, float, int]): The data for the tensor.
            requires_grad (bool): Whether or not to track gradients for this tensor.
            _children (Tuple['Tensor', ...]): Tensors that created this one (for backpropagation).
            _op (str): The operation that created this tensor.
        """
        self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def backward(self, grad: np.ndarray = None) -> None:
        """
        Perform backpropagation to compute gradients.

        Args:
            grad (np.ndarray, optional): The gradient of the loss with respect to this tensor.
                                         If None, assumes this tensor is a scalar.
        """
        if not self.requires_grad:
            return

        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be specified for non-scalar outputs")
            grad = np.ones_like(self.data)

        self.grad = grad
        topo = []
        visited = set()

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)

        build_topo(self)
        for t in reversed(topo):
            t._backward()

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def __add__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        Add another tensor or a scalar to this tensor.

        Args:
            other (Union['Tensor', float, int]): The tensor or scalar to add.

        Returns:
            Tensor: A new tensor representing the result of the addition.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='+')

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __sub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        Subtract another tensor or a scalar from this tensor.

        Args:
            other (Union['Tensor', float, int]): The tensor or scalar to subtract.

        Returns:
            Tensor: A new tensor representing the result of the subtraction.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='-')

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += -unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def exp(self) -> 'Tensor':
        """
        Compute the element-wise exponential of this tensor.

        Returns:
            Tensor: A new tensor representing the element-wise exponential of this tensor.
        """
        out = Tensor(np.exp(self.data), requires_grad=self.requires_grad,
                     _children=(self,), _op='exp')

        def _backward():
            if self.requires_grad:
                grad = unbroadcast(np.exp(self.data) * out.grad, self.data.shape)
                if self.grad is None:
                    self.grad = grad
                else:
                    self.grad += grad

        out._backward = _backward
        return out

    def log(self) -> 'Tensor':
        """
        Compute the element-wise natural logarithm of this tensor.

        Returns:
            Tensor: A new tensor representing the element-wise logarithm of this tensor.
        """
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad,
                     _children=(self,), _op='log')

        def _backward():
            if self.requires_grad:
                grad = unbroadcast((1 / self.data) * out.grad, self.data.shape)
                if self.grad is None:
                    self.grad = grad
                else:
                    self.grad += grad

        out._backward = _backward
        return out

    def softmax(self, dim: int = 1) -> 'Tensor':
        """
        Compute the softmax along the specified dimension.

        Args:
            dim (int): The dimension along which to compute the softmax.

        Returns:
            Tensor: A new tensor representing the softmax of this tensor along the specified dimension.
        """
        exps = (self - self.data.max(axis=dim, keepdims=True)).exp()
        sums = exps.data.sum(axis=dim, keepdims=True)
        out = Tensor(exps.data / sums, requires_grad=self.requires_grad,
                     _children=(self,), _op='softmax')

        def _backward():
            if self.requires_grad:
                grad = out.grad
                s = out.data
                dot = (grad * s).sum(axis=dim, keepdims=True)
                self_grad = s * (grad - dot)
                self.grad += self_grad  # safe: already same shape

        out._backward = _backward
        return out

    def cross_entropy(self, target: 'Tensor') -> 'Tensor':
        """
        Compute the cross-entropy loss.

        Args:
            target (Tensor): A tensor of true class labels.

        Returns:
            Tensor: A new tensor representing the cross-entropy loss.
        """
        probs = self.softmax(dim=1)
        N = target.data.shape[0]
        log_probs = probs.log()
        loss_vals = -log_probs.data[np.arange(N), target.data.astype(int)]
        out = Tensor(loss_vals.mean(), requires_grad=self.requires_grad,
                     _children=(self,), _op='cross_entropy')

        def _backward():
            if self.requires_grad:
                grad = probs.data.copy()
                grad[np.arange(N), target.data.astype(int)] -= 1
                grad = grad / N
                self.grad += grad

        out._backward = _backward
        return out

    def __mul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        Multiply this tensor by another tensor or scalar.

        Args:
            other (Union['Tensor', float, int]): The tensor or scalar to multiply.

        Returns:
            Tensor: A new tensor representing the result of the multiplication.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='*')

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(other.data * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __pow__(self, power: Union[int, float]) -> 'Tensor':
        """
        Raise the tensor to a power element-wise.

        Args:
            power (Union[int, float]): The exponent to raise the tensor to.

        Returns:
            Tensor: A new tensor representing the result of raising this tensor to the given power.
        """
        out = Tensor(self.data ** power, requires_grad=self.requires_grad, _children=(self,), _op=f'**{power}')

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast((power * self.data ** (power - 1)) * out.grad, self.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        return self.matmul(other)

    def matmul(self, other: 'Tensor') -> 'Tensor':
        """
        Matrix multiplication method.

        Args:
            other (Tensor): The tensor to multiply with.

        Returns:
            Tensor: A new tensor representing the result of the matrix multiplication.
        """
        assert isinstance(other, Tensor)
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='@')

        def _backward():
            if self.requires_grad:
                grad_self = out.grad @ other.data.T
                if self.grad is None:
                    self.grad = grad_self
                else:
                    self.grad += grad_self
            if other.requires_grad:
                grad_other = self.data.T @ out.grad
                if other.grad is None:
                    other.grad = grad_other
                else:
                    other.grad += grad_other

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad, _children=(self,), _op='ReLU')

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast((self.data > 0) * out.grad, self.data.shape)

        out._backward = _backward
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad,
                     _children=(self,), _op='reshape')

        def _backward():
            if self.requires_grad:
                grad = out.grad.reshape(self.data.shape)
                if self.grad is None:
                    self.grad = grad
                else:
                    self.grad += grad

        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(self.data.sum(), requires_grad=self.requires_grad, _children=(self,), _op='sum')

        def _backward():
            if self.requires_grad:
                self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out

    def mean(self):
        out = Tensor(self.data.mean(), requires_grad=self.requires_grad, _children=(self,), _op='mean')

        def _backward():
            if self.requires_grad:
                grad = (out.grad / self.data.size) * np.ones_like(self.data)
                self.grad += grad

        out._backward = _backward
        return out

    @staticmethod
    def concat(tensors: List['Tensor'], axis: int = 0) -> 'Tensor':
        data = np.concatenate([t.data for t in tensors], axis=axis)
        requires_grad = any(t.requires_grad for t in tensors)
        out = Tensor(data, requires_grad)

        def _backward():
            if not out.requires_grad:
                return

            sizes = [t.data.shape[axis] for t in tensors]
            splits = np.split(out.grad, np.cumsum(sizes[:-1]), axis=axis)

            for i, t in enumerate(tensors):
                if t.requires_grad:
                    t.grad += splits[i]

        out._backward = _backward
        out._prev = set(t for t in tensors if t.requires_grad)
        return out

    def shape(self):
        return self.data.shape

    @staticmethod
    def zeros(shape) -> 'Tensor':
        return Tensor(np.zeros(shape), requires_grad=True)

    @staticmethod
    def stack(tensors, axis=1) -> 'Tensor':
        data = np.stack([t.data for t in tensors], axis=axis)
        return Tensor(data, requires_grad=True)
