# pure-numpy-neural-network

This project is a PyTorch-style automatic differentiation engine and neural network framework built from scratch using pure NumPy. It includes tensor operations with autograd, a neural network module system (`nn.Module`-like), optimisation support, and recurrent layer implementations like LSTM.

---

## Features

- **Custom tensor class with autograd**
  - Supports basic arithmetic, matrix multiplication, softmax, cross-entropy, and activation functions.
  - Reverse-mode automatic differentiation via topological graph traversal.

- **Modular Neural Network API**
  - `Module`, `Linear`, `Sequential`, `Loss`, and activation functions.
  - Easy to compose and train neural networks.

- **LSTM Cell**
  - Single LSTM cell module with gating logic for sequence modeling.
  - Built using `Linear`, `Sigmoid`, and `Tanh`.

- **Custom Training Loop Ready**
  - Full gradient propagation with `.backward()`, and `.zero_grad()` API.

## 🔧 Core modules

### `Tensor`
- Fundamental data structure with:
  - `.backward()` for reverse-mode autodiff
  - Support for broadcasting, `exp`, `log`, `softmax`, `cross_entropy`
  - Chainable operations with gradient tracking

### `Module` (Base class)
- Abstract base for all layers
- Manages parameters and forward logic

### `Linear`
- Fully connected layer:
```python
Linear(in_features=128, out_features=64)
```
  
### `Sequential`
- Composes modules like a stack:
```python
model = Sequential(
    Linear(784, 128),
    Tanh(),
    Linear(128, 10)
)
```

### `LSTM`
- Manual LSTM logic using gates and cell states:
```python
h, c = lstm(x_t, h_prev, c_prev)
```

## 📌 To Do

- Optimiser module (SGD, Adam)
- Dataset + DataLoader utils 
- Batched LSTM/GRU modules