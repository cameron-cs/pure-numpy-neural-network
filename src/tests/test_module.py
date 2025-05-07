from src.nn import Module, Linear, Sequential
from src.tensor import Tensor
import numpy as np

def test_module_parameter_registration():
    class Dummy(Module):
        def __init__(self):
            super().__init__()
            self.w = Tensor(np.array([1.0, 2.0]), requires_grad=True)
            self._add_parameter(self.w)

        def forward(self, x):
            return x * self.w

    model = Dummy()
    assert isinstance(model.parameters(), list)
    assert len(model.parameters()) == 1
    assert model.parameters()[0] is model.w


def test_linear_forward_shape_and_values():
    in_features, out_features = 4, 3
    layer = Linear(in_features, out_features)

    # Manually override weights and bias for deterministic test
    layer.weights.data[:] = np.eye(in_features, out_features)
    layer.bias.data[:] = 1.0

    x = Tensor(np.ones((2, in_features)), requires_grad=True)
    out = layer(x)

    expected = x.data @ layer.weights.data + layer.bias.data
    assert out.data.shape == (2, out_features)
    assert np.allclose(out.data, expected)


def test_linear_backward():
    x = Tensor(np.random.randn(5, 10), requires_grad=True)
    layer = Linear(10, 3)
    out = layer(x)
    loss = out.sum()
    loss.backward()

    # Check gradients exist
    for param in layer.parameters():
        assert param.grad is not None
        assert param.grad.shape == param.data.shape
    assert x.grad.shape == x.data.shape


def test_sequential_forward():
    model = Sequential(
        Linear(4, 5),
        Linear(5, 2)
    )
    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    y = model(x)

    assert y.data.shape == (3, 2)


def test_sequential_backward():
    model = Sequential(
        Linear(4, 5),
        Linear(5, 2)
    )
    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()

    for param in model.parameters():
        assert param.grad is not None
        assert param.grad.shape == param.data.shape

    assert x.grad.shape == x.data.shape


def test_zero_grad():
    layer = Linear(3, 2)
    x = Tensor(np.ones((1, 3)), requires_grad=True)
    y = layer(x).sum()
    y.backward()

    # Ensure grads are non-zero before zero_grad
    for param in layer.parameters():
        assert not np.allclose(param.grad, 0)

    layer.zero_grad()

    # Ensure grads are zero after zero_grad
    for param in layer.parameters():
        assert np.allclose(param.grad, 0)


def test_parameters_aggregation_sequential():
    l1 = Linear(3, 4)
    l2 = Linear(4, 2)
    model = Sequential(l1, l2)
    all_params = model.parameters()

    assert isinstance(all_params, list)
    assert len(all_params) == len(l1.parameters()) + len(l2.parameters())
    assert all(p in all_params for p in l1.parameters())
    assert all(p in all_params for p in l2.parameters())