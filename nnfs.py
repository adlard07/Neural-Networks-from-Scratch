import numpy as np
from dataclasses import dataclass, field

@dataclass
class NeuralNet:
    x: np.ndarray
    y: np.ndarray

    # input size, hidden layers (including sizes) and output_size
    input_size: int
    hidden_layers: list
    output_size: int
    
    # list of activation functions
    activations: list[str] = field(default_factory=list)

    # list of weights and biases of each layer
    weights: list = field(default_factory=list)
    biases: list = field(default_factory=list)
    
    # learning rate
    learning_rate: float = 0.01

    def __post_init__(self):
        # convert y to 2D array if it's 1D
        if len(self.y.shape) == 1:
            self.y = self.y.reshape(-1, 1)
            
        # initialising list of all layers 
        self.layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        
        # initializing weights and biases with correct shapes
        for i in range(len(self.layer_sizes) - 1):
            self.weights.append(np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * np.sqrt(2.0 / self.layer_sizes[i]))
            self.biases.append(np.zeros((1, self.layer_sizes[i + 1])))

    def activation_func(self, x, activation_func):
        if activation_func == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        if activation_func == 'tanh':
            return np.tanh(x)
        if activation_func == 'relu':
            return np.maximum(0, x)
        if activation_func == 'leaky_relu':
            return np.maximum(0.01*x, x)
        
    def activation_derivative(self, x, func):
        if func == 'sigmoid':
            return x * (1 - x)
        if func == 'tanh':
            return 1 - x ** 2
        if func == 'relu':
            return np.where(x > 0, 1, 0)
        if func == 'leaky_relu':
            return np.where(x > 0, 1, 0.01)

    def neuron_activation(self, inputs, weights, bias, activation_func):
        return self.activation_func(np.dot(inputs, weights) + bias, activation_func=activation_func)

    def mean_squared_error(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    def feedforward(self):
        self.layer_outputs = [self.x]
        current_input = self.x
        
        # activation of all neurons in each layer
        for i in range(len(self.weights)):
            current_input = self.neuron_activation(
                current_input, 
                weights=self.weights[i], 
                bias=self.biases[i], 
                activation_func=self.activations[i]
            )
            self.layer_outputs.append(current_input)
        
        return current_input

    def backpropagation(self):
        # computing the feedforward output
        self.y_pred = self.feedforward()
        
        # Store the error for reporting
        error = self.mean_squared_error(self.y, self.y_pred)

        # computing the error in output layer
        self.delta = (self.y - self.y_pred) * self.activation_derivative(self.y_pred, self.activations[-1])
        
        # backpropagate through each layer
        self.deltas = [self.delta]
        for i in reversed(range(len(self.weights) - 1)):
            self.delta = np.dot(self.deltas[0], self.weights[i + 1].T) * self.activation_derivative(
                self.layer_outputs[i + 1], self.activations[i])
            self.deltas.insert(0, self.delta)
        
        # update weights & bias
        for i in range(len(self.weights)):
            self.weights[i] += np.dot(self.layer_outputs[i].T, self.deltas[i]) * self.learning_rate
            self.biases[i] += np.sum(self.deltas[i], axis=0, keepdims=True) * self.learning_rate
            
        return error

    def fit(self, epochs):
        errors = []
        for i in range(1, epochs + 1):
            error = self.backpropagation()
            errors.append(error)
            print(f"Epoch {i}/{epochs} - Error: {error:.8f}")
        return errors

    def predict(self, X):
        original_x = self.x
        self.x = X
        predictions = self.feedforward()
        self.x = original_x
        return predictions

if __name__ == "__main__":
    # test 
    x = np.array([
        [0.01510066, 0.87729392, 0.73927447],
        [0.18380298, 0.16352703, 0.23512937]])
    
    y = np.array([0.98388509, 0.76754082])

    # network configuration
    activations = ['relu', 'relu', 'sigmoid']
    hidden_layers = [4, 4] 
    
    nn = NeuralNet(
        x=x, 
        y=y, 
        input_size=x.shape[1],
        hidden_layers=hidden_layers, 
        activations=activations, 
        output_size=1,
        learning_rate=0.01
    )
    
    # training
    errors = nn.fit(epochs=100)
    
    # predictions
    predictions = nn.predict(x)
    for pred, actual in zip(predictions.flatten(), y.flatten()):
        print(f"Predicted: {pred:.8f}, Actual: {actual:.8f}")