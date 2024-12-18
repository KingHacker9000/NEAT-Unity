import math
from abc import ABC, abstractmethod

class ActivationFunction(ABC):
    """
    Abstract base class for activation functions.
    """

    @abstractmethod
    def forward(self, x):
        """
        Perform the forward pass (activation function).
        """
        pass

    @abstractmethod
    def derivative(self, x):
        """
        Compute the derivative of the activation function for backpropagation.
        """
        pass


class Sigmoid(ActivationFunction):
    """
    Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    """

    def forward(self, x):
        return 1 / (1 + math.exp(-x))

    def derivative(self, x):
        sigmoid_x = self.forward(x)
        return sigmoid_x * (1 - sigmoid_x)


class Tanh(ActivationFunction):
    """
    Tanh activation function: f(x) = tanh(x)
    """

    def forward(self, x):
        return math.tanh(x)

    def derivative(self, x):
        return 1 - math.tanh(x) ** 2


class ReLU(ActivationFunction):
    """
    ReLU activation function: f(x) = max(0, x)
    """

    def forward(self, x):
        return max(0, x)

    def derivative(self, x):
        return 1 if x > 0 else 0
    
class Linear(ActivationFunction):
    """
    Linear activation function: f(x) = x
    """

    def forward(self, x):
        return x

    def derivative(self, x):
        return 1


import math
from abc import ABC, abstractmethod

class ActivationFunction(ABC):
    """
    Abstract base class for activation functions.
    """

    @abstractmethod
    def forward(self, x):
        """
        Perform the forward pass (activation function).
        """
        pass

    @abstractmethod
    def derivative(self, x):
        """
        Compute the derivative of the activation function for backpropagation.
        """
        pass


class Sigmoid(ActivationFunction):
    """
    Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    """

    def forward(self, x):
        return 1 / (1 + math.exp(-x))

    def derivative(self, x):
        sigmoid_x = self.forward(x)
        return sigmoid_x * (1 - sigmoid_x)


class Tanh(ActivationFunction):
    """
    Tanh activation function: f(x) = tanh(x)
    """

    def forward(self, x):
        return math.tanh(x)

    def derivative(self, x):
        return 1 - math.tanh(x) ** 2


class ReLU(ActivationFunction):
    """
    ReLU activation function: f(x) = max(0, x)
    """

    def forward(self, x):
        return max(0, x)

    def derivative(self, x):
        return 1 if x > 0 else 0


def get_activation_function(function_name: str) -> ActivationFunction:
    """Return Activation Function Based on Name"""
    f = {'relu': ReLU(), 'tanh': Tanh(), 'sigmoid': Sigmoid(), 'linear': Linear()}
    assert function_name.lower() in f, f"Function Unknown: {function_name.lower()}"
    return f[function_name.lower()]

# Example usage
if __name__ == "__main__":
    values = [-2, -1, 0, 1, 2]

    sigmoid = get_activation_function('Sigmoid')
    tanh = get_activation_function('TaNh')
    relu = get_activation_function('Relu')

    print("Sigmoid:")
    for val in values:
        print(f"x: {val}, f(x): {sigmoid.forward(val):.4f}, f'(x): {sigmoid.derivative(val):.4f}")

    print("\nTanh:")
    for val in values:
        print(f"x: {val}, f(x): {tanh.forward(val):.4f}, f'(x): {tanh.derivative(val):.4f}")

    print("\nReLU:")
    for val in values:
        print(f"x: {val}, f(x): {relu.forward(val):.4f}, f'(x): {relu.derivative(val)}")
