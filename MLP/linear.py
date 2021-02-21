from .base_layer import BaseLayer
import numpy as np


class Linear(BaseLayer):
    '''Linear Neural Network layers'''
    def __init__(
        self,
        input_dims: int,
        output_dims: int
    ):
        # Initialize parameters for the linear layer randomly
        self.w = np.random.rand(input_dims, output_dims) * 0.0001
        self.b = np.random.rand(output_dims) * 0.0001
        self.dw = None
        self.db = None
        self.cache = None

    def forward(
        self,
        input_x: np.ndarray
    ):
        # Output = dot product between W and X and then add the bias
        output = np.dot(input_x, self.w) + self.b
        # Store the arrays in cache, useful for calculating the gradients in the backward pass
        self.cache = [input_x.copy(), self.w.copy(), self.b.copy()]
        return output

    def backward(self, dout):
        # dw and dx can be estimated from the incoming gradient dout, using chain rule
        temp_x, temp_w, _ = self.cache
        
        # Calculate dx
        dx = np.dot(dout, temp_w.T)
        # Calculate dw and db
        self.dw = np.dot(temp_x.T, dout)
        self.db = dout.sum(axis=0)
        # Return gradient for passing to the next layers
        return dx

    def zero_grad(self):
        # Reinitialize the gradients
        self.dw = None
        self.db = None

    @property
    def parameters(self):
        return [
            self.w,
            self.b
        ]

    @property
    def grads(self):
        return [
            self.dw,
            self.db
        ]