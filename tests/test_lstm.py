from src import nn, tensor
import numpy as np

from src.nn import LSTMCell, StackedLSTM, Module, Linear, CrossEntropyLoss
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


def validate_lstm_classifier_forward_and_backward():
    
    class LSTMClassifier(Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes):
            super().__init__()
            self.lstm = StackedLSTM(input_size, hidden_size, num_layers)
            self.fc = Linear(hidden_size, num_classes)
            self._add_parameter(self.fc.weights)
            self._add_parameter(self.fc.bias)
            for layer in self.lstm.layers:
                self._parameters.extend(layer.parameters())

        def forward(self, x_seq: Tensor) -> Tensor:
            output_seq, (h_list, _) = self.lstm(x_seq)
            final_hidden = h_list[-1]
            logits = self.fc(final_hidden)
            return logits

    np.random.seed(42)
    batch_size = 4
    seq_len = 10
    input_size = 8
    hidden_size = 16
    num_layers = 2
    num_classes = 3

    x_np = np.random.randn(batch_size, seq_len, input_size)
    y_np = np.random.randint(0, num_classes, size=(batch_size,))

    x = Tensor(x_np, requires_grad=True)
    y = Tensor(y_np)

    model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)
    criterion = CrossEntropyLoss()

    logits = model(x)
    assert logits.data.shape == (batch_size, num_classes), \
        f"Expected logits shape {(batch_size, num_classes)}, got {logits.data.shape}"

    # check argmax predictability
    preds = np.argmax(logits.data, axis=1)
    assert np.all((0 <= preds) & (preds < num_classes)), \
        f"Predicted class indices out of range: {preds}"

    loss = criterion(logits, y)
    assert np.isscalar(loss.data) or (isinstance(loss.data, np.ndarray) and loss.data.shape == ()), \
        f"Loss is not scalar: {loss.data}"

    loss.backward()

    # check gradients on final linear layer
    assert model.fc.weights.grad.shape == model.fc.weights.data.shape, \
        "Gradient shape mismatch on fc weights"
    assert not np.allclose(model.fc.weights.grad, 0), \
        "Zero gradients on fc weights — backprop may be broken"

    grad_sum = np.sum(model.fc.weights.grad, axis=0)
    assert np.all(np.abs(grad_sum) < 1e-1), \
        f"Softmax gradient sums should be small, got {grad_sum} (may vary due to small batch size)"


if __name__ == '__main__':
    test_tanh_forward_backward()
    test_sigmoid_forward_backward()
    test_lstmcell_backward()
    test_lstmcell_forward_backward()
    test_stacked_lstm()
    test_stacked_lstm_shapes_and_grads()
    test_stacked_lstm_variable_lengths()
    validate_lstm_classifier_forward_and_backward()
