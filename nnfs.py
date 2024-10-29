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
    learning_rate: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    batch_size: int = 128  # Increased batch size for better gradient estimates

    def __post_init__(self):
        if len(self.x.shape) > 2:
            self.x = self.x.reshape(self.x.shape[0], -1)
            self.input_size = self.x.shape[1]
            logging.info(f"Input data flattened to shape {self.x.shape}")
            
        self.layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        
        # Initialize weights with He initialization
        for i in range(len(self.layer_sizes) - 1):
            # He initialization for ReLU networks
            scale = np.sqrt(2.0 / self.layer_sizes[i])
            self.weights.append(np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * scale)
            self.biases.append(np.zeros((1, self.layer_sizes[i + 1])))
        
        # Initialize Adam optimizer parameters
        self.m_weights = [np.zeros_like(w) for w in self.weights]
        self.v_weights = [np.zeros_like(w) for w in self.weights]
        self.m_biases = [np.zeros_like(b) for b in self.biases]
        self.v_biases = [np.zeros_like(b) for b in self.biases]
        self.t = 0

    def get_batches(self):
        """Generate batches from the data with shuffling"""
        n_samples = len(self.x)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            yield self.x[batch_indices], self.y[batch_indices]

    def activation_func(self, inputs, activation_func):
        if activation_func == 'relu':
            return np.maximum(0, inputs)
        if activation_func == 'softmax':
            return self.softmax(inputs)
        if activation_func == 'leaky_relu':
            return np.where(inputs > 0, inputs, inputs * 0.01)
        
    def activation_derivative(self, inputs, func):
        if func == 'relu':
            return np.where(inputs > 0, 1, 0)
        if func == 'softmax':
            return 1
        if func == 'leaky_relu':
            return np.where(inputs > 0, 1, 0.01)

    def softmax(self, inputs):
        """Numerically stable softmax"""
        inputs = np.clip(inputs, -500, 500)  # Prevent overflow
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return np.clip(probabilities, 1e-7, 1 - 1e-7)  # Prevent log(0)

    def sparse_categorical_crossentropy(self, y_true, y_pred):
        """Numerically stable sparse categorical crossentropy"""
        n_samples = len(y_true)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)  # Prevent log(0)
        log_probs = -np.log(y_pred[np.arange(n_samples), y_true.astype(int)])
        return np.mean(log_probs)

    def sparse_categorical_crossentropy_gradient(self, y_true, y_pred):
        n_samples = y_pred.shape[0]
        gradient = y_pred.copy()
        gradient[np.arange(n_samples), y_true.astype(int)] -= 1
        return gradient / n_samples

    def feedforward(self, x_batch):
        """Forward pass with batch normalization"""
        self.layer_outputs = [x_batch]
        current_input = x_batch
        
        for i in range(len(self.weights)):
            # Linear transformation
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            
            # Activation
            current_input = self.activation_func(z, self.activations[i])
            
            # Store output for backprop
            self.layer_outputs.append(current_input)
        
        return current_input

    def adam_update(self, param, grad, m, v, t):
        """Adam optimizer update with gradient clipping"""
        # Gradient clipping
        grad = np.clip(grad, -1.0, 1.0)
        
        # Adam update
        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
        
        m_hat = m / (1 - self.beta1 ** t)
        v_hat = v / (1 - self.beta2 ** t)
        
        update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        param -= update
        
        return param, m, v

    def train_on_batch(self, x_batch, y_batch):
        """Train on a single batch with improved gradient computation"""
        self.t += 1
        
        # Forward pass
        y_pred = self.feedforward(x_batch)
        
        # Calculate loss
        loss = self.sparse_categorical_crossentropy(y_batch, y_pred)
        
        # Initialize gradients
        deltas = [self.sparse_categorical_crossentropy_gradient(y_batch, y_pred)]
        
        # Backpropagate through layers
        for i in reversed(range(len(self.weights) - 1)):
            delta = np.dot(deltas[0], self.weights[i + 1].T) * self.activation_derivative(
                self.layer_outputs[i + 1], self.activations[i])
            deltas.insert(0, delta)
        
        # Update weights and biases
        for i in range(len(self.weights)):
            d_weight = np.dot(self.layer_outputs[i].T, deltas[i])
            d_bias = np.sum(deltas[i], axis=0, keepdims=True)
            
            # Apply Adam updates with L2 regularization
            l2_reg = 0.0001 * self.weights[i]  # L2 regularization
            self.weights[i], self.m_weights[i], self.v_weights[i] = self.adam_update(
                self.weights[i], d_weight + l2_reg, self.m_weights[i], self.v_weights[i], self.t)
            
            self.biases[i], self.m_biases[i], self.v_biases[i] = self.adam_update(
                self.biases[i], d_bias, self.m_biases[i], self.v_biases[i], self.t)
        
        return loss

    def fit(self, epochs):
        """Train the network with early stopping"""
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        losses = []
        
        for epoch in range(1, epochs + 1):
            epoch_losses = []
            for x_batch, y_batch in self.get_batches():
                loss = self.train_on_batch(x_batch, y_batch)
                epoch_losses.append(loss)
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 1 == 0:
                logging.info(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.8f}")
            
            # Early stopping
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered at epoch {epoch}")
                break
                
        return losses

    def predict(self, X, batch_size=None):
        """Make predictions with batching"""
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        
        batch_size = batch_size or self.batch_size
        predictions = []
        
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size]
            batch_predictions = self.feedforward(batch)
            predictions.append(batch_predictions)
            
        return np.vstack(predictions)

if __name__ == "__main__":
    # Training data
    x = np.random.randn(60000, 28, 28)  # 100 samples for better testing
    y = np.random.randint(0, 10, size=60000)
    
    # Test data
    x_test = np.random.randn(10000, 28, 28)
    y_test = np.random.randint(0, 10, size=10000)
    
    # Network configuration
    activations = ['relu', 'relu', 'softmax']
    hidden_layers = [128, 64]
    
    nn = NeuralNet(
        x=x, 
        y=y, 
        input_size=784,
        hidden_layers=hidden_layers, 
        activations=activations, 
        output_size=10,
        learning_rate=0.001  # Adam typically uses lower learning rate
    )
    
    # Training
    losses = nn.fit(epochs=100)
    
    # Evaluate
    train_predictions = nn.predict(x)
    train_predicted_classes = np.argmax(train_predictions, axis=1)
    test_predictions = nn.predict(x_test)
    test_predicted_classes = np.argmax(test_predictions, axis=1)
    
    # Print results
    train_accuracy = np.mean(train_predicted_classes == y)
    test_accuracy = np.mean(test_predicted_classes == y_test)
    logging.info(f"\nFinal Results:")
    logging.info(f"Training Accuracy: {train_accuracy:.2%}")
    logging.info(f"Test Accuracy: {test_accuracy:.2%}")