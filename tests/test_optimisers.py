import numpy as np

from src.nn import Linear, MSELoss, Sequential, Tanh
from src.optim import SGDMomentum
from src.tensor import Tensor


def test_optimiser_step():
    layer = Linear(4, 2)
    optimizer = SGDMomentum(layer.parameters(), lr=0.01, momentum=0.9)

    x = Tensor(np.random.randn(1, 4), requires_grad=False)
    y = Tensor(np.random.randn(1, 2), requires_grad=False)

    loss_fn = MSELoss()

    initial_weight = layer.weights.data.copy()

    pred = layer(x)
    loss = loss_fn(pred, y)
    layer.zero_grad()
    loss.backward()
    optimizer.step()

    assert not np.allclose(initial_weight, layer.weights.data), "Weights did not update"

def test_sequential_mlp_grad():
    x = Tensor(np.random.randn(10, 3), requires_grad=False)
    y = Tensor(np.random.randn(10, 2), requires_grad=False)

    model = Sequential(
        Linear(3, 5),
        Tanh(),
        Linear(5, 2)
    )

    loss_fn = MSELoss()
    optimizer = SGDMomentum(model.parameters(), lr=0.01, momentum=0.8)

    for _ in range(50):
        out = model(x)
        loss = loss_fn(out, y)
        model.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = loss.data.item()
    assert final_loss < 1.0


def test_linear_regression_sgd_with_momentum():
    x_np = np.random.randn(100, 1)
    y_np = 2 * x_np + 1 + 0.1 * np.random.randn(100, 1)

    x = Tensor(x_np, requires_grad=False)
    y = Tensor(y_np, requires_grad=False)

    model = Linear(1, 1)
    loss_fn = MSELoss()
    optimizer = SGDMomentum(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(100):
        pred = model(x)
        loss = loss_fn(pred, y)
        model.zero_grad()
        loss.backward()
        optimizer.step()

    w = model.weights.data.item()
    b = model.bias.data.item()
    assert abs(w - 2) < 0.2 and abs(b - 1) < 0.2


if __name__ == '__main__':
    test_optimiser_step()
    test_linear_regression_sgd_with_momentum()
    test_sequential_mlp_grad()
