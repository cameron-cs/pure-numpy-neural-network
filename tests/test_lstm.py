from src import nn, tensor
import numpy as np

from src.nn import LSTMCell, StackedLSTM
from src.tensor import Tensor


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


def test_stacked_lstm():
    import numpy as np

    B, T, I, H, L = 2, 4, 3, 5, 2
    x_seq = Tensor(np.random.randn(B, T, I), requires_grad=True)

    model = StackedLSTM(input_size=I, hidden_size=H, num_layers=L)
    out, (h_list, c_list) = model(x_seq)

    assert out.data.shape == (B, T, H)
    assert all(h.data.shape == (B, H) for h in h_list)
    assert all(c.data.shape == (B, H) for c in c_list)

    # backward pass
    loss = out.sum()
    loss.backward()
    assert x_seq.grad is not None


def test_stacked_lstm_shapes_and_grads():
    import numpy as np

    B, T, I, H, L = 3, 6, 4, 7, 2  # batch=3, time=6, input=4, hidden=7, layers=2
    x_seq = Tensor(np.random.randn(B, T, I) * 0.1, requires_grad=True)

    model = StackedLSTM(input_size=I, hidden_size=H, num_layers=L)
    output_seq, (h_list, c_list) = model(x_seq)

    # shape checks
    assert output_seq.data.shape == (B, T, H), "Output seq shape mismatch"
    assert len(h_list) == L and len(c_list) == L, "Hidden/cell list length mismatch"
    for h, c in zip(h_list, c_list):
        assert h.data.shape == (B, H), "Hidden state shape mismatch"
        assert c.data.shape == (B, H), "Cell state shape mismatch"

    # Gradient check
    loss = output_seq.sum()
    loss.backward()
    assert x_seq.grad is not None, "Gradients not propagated"
    assert x_seq.grad.shape == x_seq.data.shape, "Gradient shape mismatch"


def test_stacked_lstm_variable_lengths():
    import numpy as np

    for T in [2, 5, 10]:
        B, I, H, L = 2, 3, 4, 2
        x_seq = Tensor(np.random.randn(B, T, I), requires_grad=True)
        model = StackedLSTM(input_size=I, hidden_size=H, num_layers=L)
        output_seq, _ = model(x_seq)

        assert output_seq.data.shape == (B, T, H), f"SeqLen={T}: wrong shape"
        output_seq.sum().backward()
        assert x_seq.grad is not None


if __name__ == '__main__':
    test_tanh_forward_backward()
    test_sigmoid_forward_backward()
    test_lstmcell_backward()
    test_lstmcell_forward_backward()
    test_stacked_lstm()
    test_stacked_lstm_shapes_and_grads()
    test_stacked_lstm_variable_lengths()
