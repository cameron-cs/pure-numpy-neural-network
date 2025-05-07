import numpy as np

from src import tensor

EPS = 1e-5
RTOL = 1e-4
ATOL = 1e-6


def numerical_grad(f, x: np.ndarray):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old = x[ix]
        x[ix] = old + EPS
        fxh1 = f(x)
        x[ix] = old - EPS
        fxh2 = f(x)
        grad[ix] = (fxh1 - fxh2) / (2 * EPS)
        x[ix] = old
        it.iternext()
    return grad


def test_add():
    a = tensor.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = tensor.Tensor([[10.0, 20.0], [30.0, 40.0]], requires_grad=True)
    c = a + b
    c.backward(np.ones_like(c.data))
    assert np.allclose(c.data, a.data + b.data)
    assert np.allclose(a.grad, np.ones_like(a.data))
    assert np.allclose(b.grad, np.ones_like(b.data))


def test_mul_and_pow():
    x = tensor.Tensor([2.0, 3.0], requires_grad=True)
    y = x * x
    z = y ** 2
    z.backward(np.array([1.0, 1.0]))
    # dz/dx = 4x^3
    expected = 4 * x.data ** 3
    assert np.allclose(x.grad, expected, rtol=RTOL, atol=ATOL)


def test_matmul():
    a = tensor.Tensor(np.random.randn(3, 4), requires_grad=True)
    b = tensor.Tensor(np.random.randn(4, 2), requires_grad=True)
    c = a @ b
    c.sum().backward()
    assert a.grad.shape == a.data.shape
    assert b.grad.shape == b.data.shape


def test_exp_log():
    x = tensor.Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x.exp().log()
    y.sum().backward()
    assert np.allclose(x.grad, np.ones_like(x.data), atol=ATOL)


def test_relu():
    x = tensor.Tensor([-1.0, 0.0, 1.0], requires_grad=True)
    y = x.relu()
    y.backward(np.ones_like(x.data))
    assert np.allclose(y.data, [0.0, 0.0, 1.0])
    assert np.allclose(x.grad, [0.0, 0.0, 1.0])


def test_softmax_backward():
    x = tensor.Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
    y = x.softmax()
    y.backward(np.array([[1.0, 0.0, 0.0]]))
    assert y.data.shape == x.data.shape
    assert x.grad.shape == x.data.shape


def test_cross_entropy_loss():
    x = tensor.Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    y = tensor.Tensor([2])
    loss = x.cross_entropy(y)
    loss.backward()
    assert np.isclose(loss.data, -np.log(np.exp(3) / np.sum(np.exp(x.data))), atol=ATOL)
    assert x.grad.shape == x.data.shape


def test_backward_chain():
    x = tensor.Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x * 2
    z = (y + 1).relu()
    out = z.sum()
    out.backward()
    assert x.grad.shape == x.data.shape


def test_numerical_gradient_match():
    data = np.random.randn(5)
    x = tensor.Tensor(data.copy(), requires_grad=True)
    y = (x ** 3).sum()
    y.backward()
    expected_grad = numerical_grad(lambda d: np.sum(d ** 3), data)
    assert np.allclose(x.grad, expected_grad, rtol=RTOL, atol=ATOL)


def test_broadcast_grad():
    a = tensor.Tensor(np.ones((2, 1)), requires_grad=True)
    b = tensor.Tensor(np.ones((1, 2)), requires_grad=True)
    c = a + b
    c.backward(np.ones((2, 2)))
    assert a.grad.shape == (2, 1)
    assert b.grad.shape == (1, 2)
    assert np.allclose(a.grad, [[2.0], [2.0]])
    assert np.allclose(b.grad, [[2.0, 2.0]])


if __name__ == '__main__':
    test_add()
    test_mul_and_pow()
    test_matmul()
    test_exp_log()
    test_relu()
    test_softmax_backward()
    test_cross_entropy_loss()
    test_backward_chain()
    test_numerical_gradient_match()
    test_broadcast_grad()
