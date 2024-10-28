from logger import logging
import numpy as np
from ingestion import read_data
from nnfs import NeuralNet

(x_train, y_train), (x_test, y_test) = read_data()
logging.info(f"Input shape: {x_train.shape}")

# Ensure labels are one-hot encoded
num_classes = 10
y_train_one_hot = np.zeros((len(y_train), num_classes))
y_train_one_hot[np.arange(len(y_train)), y_train] = 1

y_test_one_hot = np.zeros((len(y_test), num_classes))
y_test_one_hot[np.arange(len(y_test)), y_test] = 1

hidden_layers = [64, 128, 256, 128, 64]  # 5 hidden layers
activations = ['relu', 'relu', 'relu', 'relu', 'relu', 'softmax']

nn = NeuralNet(
    x=x_train, 
    y=y_train_one_hot, 
    input_size=784,
    hidden_layers=hidden_layers, 
    activations=activations, 
    output_size=10, 
    learning_rate=0.01
)

losses = nn.fit(epochs=100)

train_predictions = nn.predict(x_train)
train_predicted_classes = np.argmax(train_predictions, axis=1)

test_predictions = nn.predict(x_test)
test_predicted_classes = np.argmax(test_predictions, axis=1)

# Logging training results
logging.info("\nTraining Set Results:")
train_accuracy = np.mean(train_predicted_classes == y_train)
logging.info(f"Training Accuracy: {train_accuracy:.2%}")

# Logging test results
logging.info("\nTest Set Results:")
test_accuracy = np.mean(test_predicted_classes == y_test)
logging.info(f"Test Accuracy: {test_accuracy:.2%}")
 