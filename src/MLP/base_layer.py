class BaseLayer(object):
    '''
    description: abstract base class for layers
    all layer obect should have forward and backward function for the neural network
    '''
    def __init__(self):
        pass

    def forward(self, input_x):
        """
        Each Layer Should Store their own intermediate outputs for backward
        """
        pass

    def backward(self):
        """
        Use stored intermediate output and provided gradient to compute gradient
        """
        pass

    def zero_grad(self):
        pass

    def __call__(self, input_x, **kwargs):
        return self.forward(input_x, **kwargs)

    @property
    def parameters(self):
        return []

    @property
    def grads(self):
        return []
