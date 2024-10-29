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

# Note: We're no longer converting to one-hot encoding since we're using
# sparse categorical crossentropy

# Define network parameters
hidden_layers = [32, 64, 128]
activations = ['relu', 'relu', 'relu', 'softmax']

# Initialize neural network
nn = NeuralNet(
    x=x_train,
    y=y_train,
    input_size=784,
    hidden_layers=hidden_layers,
    activations=activations,
    output_size=10,
    learning_rate=0.01,
    batch_size=60  # Adjust this based on your available memory
)

# Train the network
losses = nn.fit(epochs=5)

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
correct_train_predictions = 0
for i in range(len(train_predicted_classes)):
  if  train_predicted_classes[i] == y_train[i]:
    correct_train_predictions += 1
train_accuracy = (correct_train_predictions / len(y_train) * 100)
logging.info(f"Training Accuracy: {train_accuracy:.4f}%")

# Logging test results
logging.info("\nTest Set Results:")
correct_test_predictions = 0
for i in range(len(test_predicted_classes)):
  if  test_predicted_classes[i] == y_test[i]:
    correct_test_predictions += 1
test_accuracy = (correct_test_predictions / len(y_test) * 100)
logging.info(f"Test Accuracy: {test_accuracy:.4f}%")

# Optional: Save some sample predictions for inspection
logging.info("\nSample Predictions:")
for i in range(min(10, len(y_test))):
    logging.info(f"True: {y_test[i]}, Predicted: {test_predicted_classes[i]}")