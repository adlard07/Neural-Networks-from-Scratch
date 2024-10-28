from logger import logging
import numpy as np
from ingestion import read_data
from nnfs import NeuralNet

# Load the data
(x_train, y_train), (x_test, y_test) = read_data()
logging.info(f"Input shape -> {x_train.shape}")

# Shuffle data while keeping images and labels together
train_indices = np.arange(len(x_train))
test_indices = np.arange(len(x_test))

np.random.shuffle(train_indices)
np.random.shuffle(test_indices)

x_train, y_train = x_train[train_indices], y_train[train_indices]
x_test, y_test = x_test[test_indices], y_test[test_indices]

logging.info('Data shuffled ->')

# Normalize data
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# Ensure labels are one-hot encoded for the network
num_classes = 10
y_train_one_hot = np.zeros((len(y_train), num_classes))
y_train_one_hot[np.arange(len(y_train)), y_train] = 1

y_test_one_hot = np.zeros((len(y_test), num_classes))
y_test_one_hot[np.arange(len(y_test)), y_test] = 1

# Define network parameters
hidden_layers = [32, 64, 128]
activations = ['leaky_relu', 'leaky_relu', 'leaky_relu', 'softmax']

# Reduced learning rate for better convergence
nn = NeuralNet(
    x=x_train,
    y=y_train_one_hot,
    input_size=784,
    hidden_layers=hidden_layers,
    activations=activations,
    output_size=10,
    learning_rate=0.01  # Changed to a lower learning rate
)

# Train the network
losses = nn.fit(epochs=100)

# Logging the loss values
for epoch, loss in enumerate(losses, 1):
    logging.info(f"Epoch {epoch} - Loss: {loss:.8f}")

# Training set predictions
train_predictions = nn.predict(x_train)
train_predicted_classes = np.argmax(train_predictions, axis=1)

# Test set predictions
test_predictions = nn.predict(x_test)
test_predicted_classes = np.argmax(test_predictions, axis=1)

# Logging training results
logging.info("\nTraining Set Results:")
correct_train_predictions = np.sum(train_predicted_classes == y_train)
train_accuracy = (correct_train_predictions / len(y_train)) * 100  # Correct calculation
logging.info(f"Training Accuracy: {train_accuracy:.2f}%")

# Logging test results
logging.info("\nTest Set Results:")
correct_test_predictions = np.sum(test_predicted_classes == y_test)
test_accuracy = (correct_test_predictions / len(y_test)) * 100  # Correct calculation
logging.info(f"Test Accuracy: {test_accuracy:.2f}%")
