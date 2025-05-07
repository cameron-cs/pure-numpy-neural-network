from src import nn, tensor
import numpy as np

from src.nn import LSTMCell
from src.tensor import Tensor


def test_concat_forward_backward():
    a = Tensor(np.array([[1.0, 2.0]]), requires_grad=True)
    b = Tensor(np.array([[3.0, 4.0]]), requires_grad=True)

    out = Tensor.concat([a, b], axis=0)  # shape (2,2)
    out.backward(np.ones_like(out.data))

    assert np.allclose(out.data, np.array([[1, 2], [3, 4]]))
    assert np.allclose(a.grad, np.ones_like(a.data))
    assert np.allclose(b.grad, np.ones_like(b.data))


def test_sigmoid_forward_backward():
    x = Tensor(np.array([[0.0, 2.0]]), requires_grad=True)
    y = nn.Sigmoid().forward(x)
    y.backward(np.ones_like(y.data))

    expected = 1 / (1 + np.exp(-x.data))
    expected_grad = expected * (1 - expected)

    assert np.allclose(y.data, expected)
    assert np.allclose(x.grad, expected_grad)


def test_tanh_forward_backward():
    x = Tensor(np.array([[0.0, 1.0]]), requires_grad=True)
    y = nn.Tanh().forward(x)
    y.backward(np.ones_like(y.data))

    expected = np.tanh(x.data)
    expected_grad = 1 - expected ** 2

    assert np.allclose(y.data, expected)
    assert np.allclose(x.grad, expected_grad)


def test_lstmcell_backward():
    input_size = 2
    hidden_size = 3
    batch_size = 1

    lstm = LSTMCell(input_size, hidden_size)

    x = Tensor(np.random.randn(batch_size, input_size), requires_grad=True)
    h_prev = Tensor(np.zeros((batch_size, hidden_size)), requires_grad=True)
    c_prev = Tensor(np.zeros((batch_size, hidden_size)), requires_grad=True)

    h, c = lstm(x, h_prev, c_prev)

    loss = h.sum()
    loss.backward()

    assert x.grad is not None
    assert h_prev.grad is not None
    assert c_prev.grad is not None


def test_lstmcell_forward_backward():
    batch_size, input_size, hidden_size = 2, 3, 4

    lstm = nn.LSTMCell(input_size, hidden_size)

    x = Tensor(np.random.randn(batch_size, input_size), requires_grad=True)
    h = Tensor(np.zeros((batch_size, hidden_size)), requires_grad=True)
    c = Tensor(np.zeros((batch_size, hidden_size)), requires_grad=True)

    h_new, c_new = lstm(x, h, c)

    # forward checks
    assert h_new.data.shape == (batch_size, hidden_size)
    assert c_new.data.shape == (batch_size, hidden_size)

    # dummy loss: sum of all elements in h_new
    loss = h_new.sum()
    loss.backward()

    # gradient checks
    assert x.grad.shape == x.data.shape
    assert h.grad.shape == h.data.shape
    assert c.grad.shape == c.data.shape

    # gradient non-zero check
    assert np.any(x.grad != 0), "x.grad is zero"
    assert np.any(h.grad != 0), "h.grad is zero"
    assert np.any(c.grad != 0), "c.grad is zero"


if __name__ == '__main__':
    test_tanh_forward_backward()
    test_sigmoid_forward_backward()
    test_concat_forward_backward()
    test_lstmcell_backward()
    test_lstmcell_forward_backward()
