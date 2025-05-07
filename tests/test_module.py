from src import nn, tensor
import numpy as np


def test_module_parameter_registration():
    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = tensor.Tensor(np.array([1.0, 2.0]), requires_grad=True)
            self._add_parameter(self.w)

        def forward(self, x):
            return x * self.w

    model = Dummy()
    assert isinstance(model.parameters(), list)
    assert len(model.parameters()) == 1
    assert model.parameters()[0] is model.w


def test_linear_forward_shape_and_values():
    in_features, out_features = 4, 3
    layer = nn.Linear(in_features, out_features)

    # Manually override weights and bias for deterministic test
    layer.weights.data[:] = np.eye(in_features, out_features)
    layer.bias.data[:] = 1.0

    x = tensor.Tensor(np.ones((2, in_features)), requires_grad=True)
    out = layer(x)

    expected = x.data @ layer.weights.data + layer.bias.data
    assert out.data.shape == (2, out_features)
    assert np.allclose(out.data, expected)


def test_linear_backward():
    x = tensor.Tensor(np.random.randn(5, 10), requires_grad=True)
    layer = nn.Linear(10, 3)
    out = layer(x)
    loss = out.sum()
    loss.backward()

    # Check gradients exist
    for param in layer.parameters():
        assert param.grad is not None
        assert param.grad.shape == param.data.shape
    assert x.grad.shape == x.data.shape


def test_sequential_forward():
    model = nn.Sequential(
        nn.Linear(4, 5),
        nn.Linear(5, 2)
    )
    x = tensor.Tensor(np.random.randn(3, 4), requires_grad=True)
    y = model(x)

    assert y.data.shape == (3, 2)


def test_sequential_backward():
    model = nn.Sequential(
        nn.Linear(4, 5),
        nn.Linear(5, 2)
    )
    x = tensor.Tensor(np.random.randn(3, 4), requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()

    for param in model.parameters():
        assert param.grad is not None
        assert param.grad.shape == param.data.shape

    assert x.grad.shape == x.data.shape


def test_zero_grad():
    layer = nn.Linear(3, 2)
    x = tensor.Tensor(np.ones((1, 3)), requires_grad=True)
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
    l1 = nn.Linear(3, 4)
    l2 = nn.Linear(4, 2)
    model = nn.Sequential(l1, l2)
    all_params = model.parameters()

    assert isinstance(all_params, list)
    assert len(all_params) == len(l1.parameters()) + len(l2.parameters())
    assert all(p in all_params for p in l1.parameters())
    assert all(p in all_params for p in l2.parameters())


def test_linear_regression_simple():
    np.random.seed(0)

    # Create synthetic data: y = 2 * x + 3
    x_data = np.random.randn(100, 1)
    y_data = 2 * x_data + 3

    x = tensor.Tensor(x_data, requires_grad=False)
    y_true = tensor.Tensor(y_data, requires_grad=False)

    # Model: Linear layer with 1 input, 1 output
    model = nn.Linear(1, 1)
    loss_fn = nn.MSELoss()
    lr = 0.1

    # Train for a few epochs
    for epoch in range(100):
        y_pred = model(x)
        loss = loss_fn(y_pred, y_true)

        model.zero_grad()
        loss.backward()

        # Manual SGD
        for param in model.parameters():
            param.data -= lr * param.grad

    # Test learned parameters are close to [2.0, 3.0]
    w_learned = model.weights.data.item()
    b_learned = model.bias.data.item()

    assert abs(w_learned - 2.0) < 0.1, f"Weight too far: {w_learned}"
    assert abs(b_learned - 3.0) < 0.1, f"Bias too far: {b_learned}"


def validate_logistic_regression_forward_and_backward():
    class LogisticRegression(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)
            self.sigmoid = nn.Sigmoid()
            self._add_parameter(self.linear.weights)
            self._add_parameter(self.linear.bias)

        def forward(self, x: tensor.Tensor) -> tensor.Tensor:
            x = self.linear(x)
            logits = self.sigmoid(x)
            return logits  # sigmoid activation for binary classification

    np.random.seed(42)
    input_dim = 5
    batch_size = 6

    # fake data gen
    X_np = np.random.randn(batch_size, input_dim)
    y_np = np.random.randint(0, 2, size=(batch_size, 1))  # binary targets: 0 or 1

    X = tensor.Tensor(X_np, requires_grad=True)
    y = tensor.Tensor(y_np.astype(np.float32))  # match dtype for numerical ops

    model = LogisticRegression(input_dim)
    criterion = nn.BCELoss()

    preds = model(X)

    assert preds.data.shape == (batch_size, 1), \
        f"Expected shape {(batch_size, 1)}, got {preds.data.shape}"
    assert np.all((0 <= preds.data) & (preds.data <= 1)), \
        f"Sigmoid output should be in [0,1], got {preds.data}"

    loss = criterion(preds, y)
    assert np.isscalar(loss.data) or (isinstance(loss.data, np.ndarray) and loss.data.shape == ()), \
        f"Loss is not scalar: {loss.data}"

    loss.backward()

    # gradient check
    assert model.linear.weights.grad.shape == model.linear.weights.data.shape, \
        "Gradient shape mismatch on weights"
    assert not np.allclose(model.linear.weights.grad, 0), \
        "Zero gradient — backprop may be broken"


def test_logistic_regression_with_bce():
    class LogisticRegression(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)
            self.sigmoid = nn.Sigmoid()
            self._add_parameter(self.linear.weights)
            self._add_parameter(self.linear.bias)

        def forward(self, x: tensor.Tensor) -> tensor.Tensor:
            x = self.linear(x)
            return self.sigmoid(x)  # outputs shape (batch, 1)

    np.random.seed(42)

    # gen toy binary classification data
    num_samples = 10
    input_dim = 5

    x_np = np.random.randn(num_samples, input_dim)
    true_weights = np.random.randn(input_dim)
    logits_np = x_np @ true_weights
    y_np = (logits_np > 0).astype(np.float32)  # convert to 0/1

    x = tensor.Tensor(x_np, requires_grad=True)
    y = tensor.Tensor(y_np.reshape(-1, 1))  # shape (N, 1)

    model = LogisticRegression(input_dim)
    criterion = nn.BCELoss()

    # Forward
    probs = model(x)  # shape (N, 1)
    assert probs.data.shape == (num_samples, 1), "Output shape mismatch"
    assert np.all((0 < probs.data) & (probs.data < 1)), "Sigmoid probs out of bounds"

    loss = criterion(probs, y)
    assert np.isscalar(loss.data) or (isinstance(loss.data, np.ndarray) and loss.data.shape == ()), \
        f"Loss is not scalar: {loss.data}"

    loss.backward()

    # check gradients
    assert model.linear.weights.grad.shape == model.linear.weights.data.shape, \
        "Gradient shape mismatch on weights"
    assert not np.allclose(model.linear.weights.grad, 0), \
        "Zero gradients on weights — backward may be broken"
    assert model.linear.bias.grad.shape == model.linear.bias.data.shape, \
        "Gradient shape mismatch on bias"


def test_softmax_classifier():
    class SoftmaxClassifier(nn.Module):
        def __init__(self, in_features, num_classes):
            super().__init__()
            self.linear = nn.Linear(in_features, num_classes)
            self.softmax = nn.Softmax(axis=1)

            self._add_parameter(self.linear.weights)
            self._add_parameter(self.linear.bias)

        def forward(self, x: tensor.Tensor) -> tensor.Tensor:
            logits = self.linear(x)
            probs = self.softmax(logits)
            return probs

    np.random.seed(42)
    batch_size = 5
    in_features = 4
    num_classes = 3

    x_np = np.random.randn(batch_size, in_features)
    y_np = np.random.randint(0, num_classes, size=(batch_size,))

    x = tensor.Tensor(x_np, requires_grad=True)
    y = tensor.Tensor(y_np)

    model = SoftmaxClassifier(in_features, num_classes)
    probs = model(x)

    # probabilities must sum to 1 per row
    row_sums = np.sum(probs.data, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), f"Row probs don't sum to 1: {row_sums}"

    # compute log-probs and NLL manually
    log_probs = probs.log()
    loss = nn.NLLLoss()(log_probs, y)

    assert np.isscalar(loss.data) or loss.data.shape == (), f"Loss not scalar: {loss.data}"

    loss.backward()

    # check gradients
    assert model.linear.weights.grad is not None, "No gradient on weights"
    assert not np.allclose(model.linear.weights.grad, 0), "Weights gradient is zero"


if __name__ == '__main__':
    test_module_parameter_registration()
    test_linear_forward_shape_and_values()
    test_linear_backward()
    test_sequential_forward()
    test_sequential_backward()
    test_zero_grad()
    test_parameters_aggregation_sequential()
    test_linear_regression_simple()
    validate_logistic_regression_forward_and_backward()
    test_logistic_regression_with_bce()
    test_softmax_classifier()
