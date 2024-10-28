from logger import logging
import numpy as np
from dataclasses import dataclass, field

@dataclass
class NeuralNet:
    x: np.ndarray
    y: np.ndarray

    input_size: int
    hidden_layers: list
    output_size: int
    
    activations: list[str] = field(default_factory=list)
    weights: list = field(default_factory=list)
    biases: list = field(default_factory=list)
    learning_rate: float = 0.01

    def __post_init__(self):
        # Flatten the input if it's multi-dimensional
        if len(self.x.shape) > 2:
            self.x = self.x.reshape(self.x.shape[0], -1)
            # Update input size to match flattened dimension
            self.input_size = self.x.shape[1]
            logging.info(f"Input data flattened to shape {self.x.shape}")

        # convert y to one-hot encoding if needed
        if len(self.y.shape) == 1 or self.y.shape[1] == 1:
            num_classes = max(self.y.flatten()) + 1
            y_one_hot = np.zeros((len(self.y), num_classes))
            y_one_hot[np.arange(len(self.y)), self.y.flatten().astype(int)] = 1
            self.y = y_one_hot
            
        # update output size to match y shape if needed
        if self.output_size != self.y.shape[1]:
            self.output_size = self.y.shape[1]
            logging.info(f"Output size adjusted to {self.output_size} to match target shape")
            
        # initialising list of all layers 
        self.layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        
        # initializing weights and biases with correct shapes
        for i in range(len(self.layer_sizes) - 1):
            self.weights.append(np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * np.sqrt(2.0 / self.layer_sizes[i]))
            self.biases.append(np.zeros((1, self.layer_sizes[i + 1])))
            
        logging.info("Weights and biases initialized!")

    def activation_func(self, inputs, activation_func):
        if activation_func == 'sigmoid':
            return 1 / (1 + np.exp(-inputs))
        if activation_func == 'softmax':
            exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
            return exp_values / np.sum(exp_values, axis=1, keepdims=True)
        if activation_func == 'tanh':
            return np.tanh(inputs)
        if activation_func == 'relu':
            return np.maximum(0, inputs)
        if activation_func == 'leaky_relu':
            return np.maximum(0.01 * inputs, inputs)
        
    def activation_derivative(self, inputs, func):
        if func == 'sigmoid':
            return inputs * (1 - inputs)
        if func == 'softmax':
            return 1
        if func == 'tanh':
            return 1 - inputs ** 2
        if func == 'relu':
            return np.where(inputs > 0, 1, 0)
        if func == 'leaky_relu':
            return np.where(inputs > 0, 1, 0.01)

    def neuron_activation(self, inputs, weights, bias, activation_func):
        return self.activation_func(np.dot(inputs, weights) + bias, activation_func=activation_func)

    def cross_entropy_loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def feedforward(self):
        self.layer_outputs = [self.x]
        current_input = self.x
        
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
        self.y_pred = self.feedforward()
        loss = self.cross_entropy_loss(self.y, self.y_pred)
        
        batch_size = self.y.shape[0]
        self.delta = (self.y_pred - self.y) / batch_size
        
        self.deltas = [self.delta]
        for i in reversed(range(len(self.weights) - 1)):
            self.delta = np.dot(self.deltas[0], self.weights[i + 1].T) * self.activation_derivative(
                self.layer_outputs[i + 1], self.activations[i])
            self.deltas.insert(0, self.delta)
        
        for i in range(len(self.weights)):
            self.weights[i] += -self.learning_rate * np.dot(self.layer_outputs[i].T, self.deltas[i])
            self.biases[i] += -self.learning_rate * np.sum(self.deltas[i], axis=0, keepdims=True)
            
        return loss

    def fit(self, epochs):
        losses = []
        for i in range(1, epochs + 1):
            loss = self.backpropagation()
            losses.append(loss)
            if i % 10 == 0:
                logging.info(f"Epoch {i}/{epochs} - Loss: {loss:.8f}")
        return losses

    def predict(self, X):
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
            
        original_x = self.x
        self.x = X
        predictions = self.feedforward()
        self.x = original_x
        return predictions


if __name__ == "__main__":
    # Training data
    x = np.random.randn(10, 28, 28)  # 10 samples, 28x28 images
    y = np.random.randint(0, 10, size=10)  # 10 samples, 10 possible classes
    
    # Test data
    x_test = np.random.randn(10, 28, 28)  # 10 test samples
    y_test = np.random.randint(0, 10, size=10)  # 10 test labels
    
    # network configuration
    activations = ['relu', 'relu', 'softmax']
    hidden_layers = [128, 64]  # hidden layer sizes
    
    nn = NeuralNet(
        x=x, 
        y=y, 
        input_size=784,  # 28*28 = 784 (flattened input)
        hidden_layers=hidden_layers, 
        activations=activations, 
        output_size=10,  # 10 classes
        learning_rate=0.01
    )
    
    # Training
    losses = nn.fit(epochs=100)
    
    # Training set predictions
    train_predictions = nn.predict(x)
    train_predicted_classes = np.argmax(train_predictions, axis=1)
    
    # Test set predictions
    test_predictions = nn.predict(x_test)
    test_predicted_classes = np.argmax(test_predictions, axis=1)
    
    # logging.info training results
    logging.info("\nTraining Set Results:")
    train_accuracy = np.mean(train_predicted_classes == y)
    logging.info(f"Training Accuracy: {train_accuracy:.2%}")
    for i, (pred, actual) in enumerate(zip(train_predicted_classes, y)):
        logging.info(f"Sample {i + 1} - Predicted: {pred}, Actual: {actual}")
    
    # logging.info test results
    logging.info("\nTest Set Results:")
    test_accuracy = np.mean(test_predicted_classes == y_test)
    logging.info(f"Test Accuracy: {test_accuracy:.2%}")
    for i, (pred, actual) in enumerate(zip(test_predicted_classes, y_test)):
        logging.info(f"Sample {i + 1} - Predicted: {pred}, Actual: {actual}")