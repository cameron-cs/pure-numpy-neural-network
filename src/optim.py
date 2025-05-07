import numpy as np


class Optimiser:
    """
    Base class for all optimisers.
    """

    def __init__(self, params):
        """
        Args:
            params (list of Tensor): Parameters to optimise.
        """
        self._params = params

    def parameters(self):
        """
        Returns the list of parameters.
        """
        return self._params

    def step(self):
        """
        Performs a single optimisation step.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("`step()` must be implemented in subclass.")


class SGDMomentum(Optimiser):

    def __init__(self, params, lr=0.01, momentum=0.9):
        """
        Initializes the SGD optimiser with momentum.

        Args:
            params (iterable): List of parameters (weights and biases) to optimise.
            lr (float): Learning rate. Defaults to 0.01.
            momentum (float): Momentum factor. Defaults to 0.9.
        """
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.velocity = [np.zeros_like(p.data) for p in self.parameters()]

    def step(self):
        """
        Performs a single optimisation step (updates parameters).
        """
        for i, param in enumerate(self.parameters()):
            if param.requires_grad:
                # update the velocity with momentum
                self.velocity[i] = self.momentum * self.velocity[i] + param.grad
                # update the parameters
                param.data -= self.lr * self.velocity[i]
