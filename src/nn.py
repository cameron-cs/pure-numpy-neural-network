from typing import Tuple, Optional, List

import numpy as np

from src.nn_utils import xavier_init
from src.tensor import Tensor


class Module:
    """
    The base class for all neural network modules (layers).
    Provides a forward method and parameter management.
    """

    def __init__(self):
        self.requires_grad = True
        self._parameters = []

    def forward(self, *input: Tensor) -> Tensor:
        """
        The forward pass for this module.
        Must be overridden by child classes.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def zero_grad(self):
        """
        Zero the gradients of all parameters.
        """
        for param in self.parameters():
            param.zero_grad()

    def parameters(self):
        """
        Return a list of parameters (weights and biases) of the module.
        """
        return self._parameters

    def _add_parameter(self, param: Tensor):
        """
        Add a tensor to the list of parameters.
        This is a helper method used by subclasses.
        """
        self._parameters.append(param)


class Linear(Module):
    """
    Fully connected layer (Linear layer).
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weights = Tensor(xavier_init((in_features, out_features)), requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

        self._add_parameter(self.weights)
        self._add_parameter(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass through the linear layer.
        """
        return input.matmul(self.weights) + self.bias


class Sequential(Module):
    """
    A container for layers arranged sequentially.
    """

    def __init__(self, *layers: Module):
        super().__init__()
        self.layers = layers

    def forward(self, input: Tensor) -> Tensor:
        """
        Pass input through the sequential stack of layers.
        """
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def parameters(self):
        """
        Returns all the parameters from the contained layers.
        """
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


class MSELoss(Module):

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # calculate the difference between predictions and true values
        diff = y_pred - y_true
        # compute mean squared error (MSE)
        return (diff * diff).mean()


class Sigmoid(Module):
    """
    Sigmoid activation function module.

    Applies the element-wise sigmoid function:
        σ(x) = 1 / (1 + exp(-x))

    Backward pass computes the gradient:
        dL/dx = σ(x) * (1 - σ(x)) * dL/dout
    """

    def forward(self, x: Tensor) -> Tensor:
        out_data = 1 / (1 + np.exp(-x.data))
        out = Tensor(out_data, requires_grad=x.requires_grad)

        def _backward():
            if x.requires_grad:
                grad = out.data * (1 - out.data) * out.grad
                x.grad += grad

        out._backward = _backward
        out._prev = {x}
        return out


class Tanh(Module):
    """
    Tanh activation function module.

    Applies the element-wise hyperbolic tangent:
        tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    Backward pass computes the gradient:
        dL/dx = (1 - tanh(x)^2) * dL/dout
    """

    def forward(self, x: Tensor) -> Tensor:
        out_data = np.tanh(x.data)
        out = Tensor(out_data, requires_grad=x.requires_grad)

        def _backward():
            if x.requires_grad:
                grad = (1 - out.data ** 2) * out.grad
                x.grad += grad

        out._backward = _backward
        out._prev = {x}
        return out


class LSTMCell(Module):
    """
    A single LSTM cell.

    Inputs:
        - x: Input tensor at current timestep (batch_size, input_size)
        - h_prev: Hidden state from previous timestep (batch_size, hidden_size)
        - c_prev: Cell state from previous timestep (batch_size, hidden_size)

    Output:
        - h: Updated hidden state (batch_size, hidden_size)
        - c: Updated cell state (batch_size, hidden_size)

    Internals:
        - Four linear layers for gates: input (i), forget (f), output (o), cell candidate (g)
        - Activation: sigmoid for gates, tanh for candidate and final output
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.W_i = Linear(input_size + hidden_size, hidden_size)
        self.W_f = Linear(input_size + hidden_size, hidden_size)
        self.W_o = Linear(input_size + hidden_size, hidden_size)
        self.W_c = Linear(input_size + hidden_size, hidden_size)

        self.sigmoid = Sigmoid()
        self.tanh = Tanh()

    def forward(self, x: Tensor, h_prev: Tensor, c_prev: Tensor) -> Tuple[Tensor, Tensor]:
        combined = Tensor.concat([x, h_prev], axis=1)

        i = self.sigmoid(self.W_i(combined))  # input gate
        f = self.sigmoid(self.W_f(combined))  # forget gate
        o = self.sigmoid(self.W_o(combined))  # output gate
        g = self.tanh(self.W_c(combined))  # candidate cell

        c = f * c_prev + i * g  # new cell state
        h = o * self.tanh(c)  # new hidden state
        return h, c


class StackedLSTM(Module):
    """
    A stack of LSTM cells.

    Inputs:
        - x_seq: (batch_size, seq_len, input_size)
        - h0: Optional list of hidden states for each layer [(batch, hidden), ...]
        - c0: Optional list of cell states for each layer [(batch, hidden), ...]

    Outputs:
        - output_seq: (batch_size, seq_len, hidden_size)
        - (h_list, c_list): final states for each layer
    """

    def __init__(self, input_size, hidden_size, num_layers, **cell_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layers = []
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            cell = LSTMCell(layer_input_size, hidden_size, **cell_kwargs)
            self.layers.append(cell)

    def forward(self, x_seq, h0=None, c0=None):
        batch_size, seq_len, _ = x_seq.shape()

        if h0 is None:
            h0 = [Tensor.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers)]
        if c0 is None:
            c0 = [Tensor.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers)]

        outputs = []
        h_list = h0
        c_list = c0

        for t in range(seq_len):
            x_t = Tensor(x_seq.data[:, t, :], requires_grad=True)

            for layer_idx, cell in enumerate(self.layers):
                h_prev = h_list[layer_idx]
                c_prev = c_list[layer_idx]
                h, c = cell(x_t, h_prev, c_prev)
                h_list[layer_idx] = h
                c_list[layer_idx] = c
                x_t = h

            outputs.append(x_t)

        output_seq = Tensor.stack(outputs, axis=1)
        return output_seq, (h_list, c_list)
