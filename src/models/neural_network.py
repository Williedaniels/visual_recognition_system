"""
Neural Network Implementation from Scratch
Built for Visual Recognition System

This module implements neural networks from scratch using only NumPy,
including forward propagation, backpropagation, and various layer types.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Callable, Dict, Any
import pickle
import json


class ActivationFunction:
    """Collection of activation functions and their derivatives."""
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid function."""
        s = ActivationFunction.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU function."""
        return (x > 0).astype(float)
    
    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Leaky ReLU activation function."""
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Derivative of Leaky ReLU function."""
        return np.where(x > 0, 1, alpha)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Tanh activation function."""
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of tanh function."""
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Softmax activation function."""
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class Layer:
    """Base class for neural network layers."""
    
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass through the layer."""
        raise NotImplementedError
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """Backward pass through the layer."""
        raise NotImplementedError


class DenseLayer(Layer):
    """Fully connected (dense) layer."""
    
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # Initialize weights using Xavier initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))
        
        # For momentum and Adam optimizer
        self.weights_momentum = np.zeros_like(self.weights)
        self.biases_momentum = np.zeros_like(self.biases)
        self.weights_velocity = np.zeros_like(self.weights)
        self.biases_velocity = np.zeros_like(self.biases)
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass through dense layer."""
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.biases
        return self.output
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """Backward pass through dense layer."""
        # Compute gradients
        weights_gradient = np.dot(self.input.T, output_gradient)
        biases_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        input_gradient = np.dot(output_gradient, self.weights.T)
        
        # Update weights and biases
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient
        
        return input_gradient
    
    def backward_with_momentum(self, output_gradient: np.ndarray, learning_rate: float, 
                              momentum: float = 0.9) -> np.ndarray:
        """Backward pass with momentum optimization."""
        # Compute gradients
        weights_gradient = np.dot(self.input.T, output_gradient)
        biases_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        input_gradient = np.dot(output_gradient, self.weights.T)
        
        # Update momentum
        self.weights_momentum = momentum * self.weights_momentum + learning_rate * weights_gradient
        self.biases_momentum = momentum * self.biases_momentum + learning_rate * biases_gradient
        
        # Update weights and biases
        self.weights -= self.weights_momentum
        self.biases -= self.biases_momentum
        
        return input_gradient


class ActivationLayer(Layer):
    """Activation layer."""
    
    def __init__(self, activation_function: Callable, activation_derivative: Callable):
        super().__init__()
        self.activation = activation_function
        self.activation_derivative = activation_derivative
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass through activation layer."""
        self.input = input_data
        self.output = self.activation(input_data)
        return self.output
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """Backward pass through activation layer."""
        return output_gradient * self.activation_derivative(self.input)


class ConvolutionalLayer(Layer):
    """2D Convolutional layer (simplified implementation)."""
    
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize kernels
        self.kernels = np.random.randn(output_channels, input_channels, kernel_size, kernel_size) * 0.1
        self.biases = np.zeros(output_channels)
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass through convolutional layer."""
        self.input = input_data
        batch_size, channels, height, width = input_data.shape
        
        # Calculate output dimensions
        output_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, self.output_channels, output_height, output_width))
        
        # Apply padding if needed
        if self.padding > 0:
            padded_input = np.pad(input_data, 
                                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 
                                mode='constant')
        else:
            padded_input = input_data
        
        # Perform convolution
        for b in range(batch_size):
            for oc in range(self.output_channels):
                for y in range(0, output_height):
                    for x in range(0, output_width):
                        y_start = y * self.stride
                        x_start = x * self.stride
                        
                        # Extract patch
                        patch = padded_input[b, :, 
                                           y_start:y_start + self.kernel_size,
                                           x_start:x_start + self.kernel_size]
                        
                        # Compute convolution
                        output[b, oc, y, x] = np.sum(patch * self.kernels[oc]) + self.biases[oc]
        
        self.output = output
        return output
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """Backward pass through convolutional layer (simplified)."""
        # This is a simplified implementation
        # In practice, you'd implement proper gradient computation for convolution
        batch_size, channels, height, width = self.input.shape
        input_gradient = np.zeros_like(self.input)
        
        # Update kernels and biases (simplified)
        kernel_gradient = np.zeros_like(self.kernels)
        bias_gradient = np.sum(output_gradient, axis=(0, 2, 3))
        
        self.kernels -= learning_rate * kernel_gradient
        self.biases -= learning_rate * bias_gradient
        
        return input_gradient


class LossFunction:
    """Collection of loss functions and their derivatives."""
    
    @staticmethod
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean squared error loss."""
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Derivative of mean squared error."""
        return 2 * (y_pred - y_true) / y_true.size
    
    @staticmethod
    def categorical_crossentropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Categorical crossentropy loss."""
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    @staticmethod
    def crossentropy_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Derivative of categorical crossentropy."""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (y_pred - y_true) / y_true.shape[0]


class NeuralNetwork:
    """Neural network class that combines layers."""
    
    def __init__(self):
        self.layers = []
        self.loss_function = None
        self.loss_derivative = None
        self.training_history = {'loss': [], 'accuracy': []}
    
    def add_layer(self, layer: Layer):
        """Add a layer to the network."""
        self.layers.append(layer)
    
    def set_loss(self, loss_function: Callable, loss_derivative: Callable):
        """Set the loss function."""
        self.loss_function = loss_function
        self.loss_derivative = loss_derivative
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass through the entire network."""
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float):
        """Backward pass through the entire network."""
        gradient = output_gradient
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int, 
              learning_rate: float = 0.01, batch_size: int = 32, 
              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the neural network.
        
        Args:
            X_train: Training input data
            y_train: Training target data
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            batch_size: Size of training batches
            validation_data: Optional validation data (X_val, y_val)
            verbose: Whether to print training progress
            
        Returns:
            Training history dictionary
        """
        n_samples = X_train.shape[0]
        n_batches = n_samples // batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_accuracy = 0
            
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            for batch in range(n_batches):
                # Get batch data
                start_idx = batch * batch_size
                end_idx = start_idx + batch_size
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                predictions = self.forward(X_batch)
                
                # Compute loss
                loss = self.loss_function(y_batch, predictions)
                epoch_loss += loss
                
                # Compute accuracy
                if len(y_batch.shape) > 1 and y_batch.shape[1] > 1:
                    # Multi-class classification
                    pred_classes = np.argmax(predictions, axis=1)
                    true_classes = np.argmax(y_batch, axis=1)
                    accuracy = np.mean(pred_classes == true_classes)
                else:
                    # Binary classification or regression
                    accuracy = np.mean(np.abs(predictions - y_batch) < 0.5)
                
                epoch_accuracy += accuracy
                
                # Backward pass
                gradient = self.loss_derivative(y_batch, predictions)
                self.backward(gradient, learning_rate)
            
            # Average metrics over batches
            avg_loss = epoch_loss / n_batches
            avg_accuracy = epoch_accuracy / n_batches
            
            self.training_history['loss'].append(avg_loss)
            self.training_history['accuracy'].append(avg_accuracy)
            
            # Validation
            if validation_data is not None:
                X_val, y_val = validation_data
                val_predictions = self.forward(X_val)
                val_loss = self.loss_function(y_val, val_predictions)
                
                if len(y_val.shape) > 1 and y_val.shape[1] > 1:
                    val_pred_classes = np.argmax(val_predictions, axis=1)
                    val_true_classes = np.argmax(y_val, axis=1)
                    val_accuracy = np.mean(val_pred_classes == val_true_classes)
                else:
                    val_accuracy = np.mean(np.abs(val_predictions - y_val) < 0.5)
            
            if verbose and (epoch + 1) % 10 == 0:
                if validation_data is not None:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Loss: {avg_loss:.4f} - Acc: {avg_accuracy:.4f} - "
                          f"Val Loss: {val_loss:.4f} - Val Acc: {val_accuracy:.4f}")
                else:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Loss: {avg_loss:.4f} - Accuracy: {avg_accuracy:.4f}")
        
        return self.training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input data."""
        return self.forward(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        """Evaluate the model on test data."""
        predictions = self.predict(X_test)
        loss = self.loss_function(y_test, predictions)
        
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            pred_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(y_test, axis=1)
            accuracy = np.mean(pred_classes == true_classes)
        else:
            accuracy = np.mean(np.abs(predictions - y_test) < 0.5)
        
        return loss, accuracy
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        model_data = {
            'layers': [],
            'training_history': self.training_history
        }
        
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer_data = {
                    'type': 'dense',
                    'weights': layer.weights.tolist(),
                    'biases': layer.biases.tolist(),
                    'input_size': layer.input_size,
                    'output_size': layer.output_size
                }
            elif isinstance(layer, ActivationLayer):
                layer_data = {
                    'type': 'activation',
                    'function': layer.activation.__name__ if hasattr(layer.activation, '__name__') else 'custom'
                }
            else:
                layer_data = {'type': 'unknown'}
            
            model_data['layers'].append(layer_data)
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def plot_training_history(self):
        """Plot training history."""
        if not self.training_history['loss']:
            print("No training history to plot.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(self.training_history['loss'])
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.training_history['accuracy'])
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()


def create_simple_classifier(input_size: int, hidden_size: int, output_size: int) -> NeuralNetwork:
    """
    Create a simple neural network classifier.
    
    Args:
        input_size: Size of input layer
        hidden_size: Size of hidden layer
        output_size: Size of output layer
        
    Returns:
        Configured neural network
    """
    network = NeuralNetwork()
    
    # Add layers
    network.add_layer(DenseLayer(input_size, hidden_size))
    network.add_layer(ActivationLayer(ActivationFunction.relu, ActivationFunction.relu_derivative))
    network.add_layer(DenseLayer(hidden_size, output_size))
    network.add_layer(ActivationLayer(ActivationFunction.softmax, lambda x: x))  # Softmax derivative handled in loss
    
    # Set loss function
    network.set_loss(LossFunction.categorical_crossentropy, LossFunction.crossentropy_derivative)
    
    return network


def test_neural_network():
    """Test function for the neural network."""
    print("Testing Neural Network...")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    n_classes = 3
    
    # Generate random data
    X = np.random.randn(n_samples, n_features)
    
    # Create synthetic labels
    y_indices = np.random.randint(0, n_classes, n_samples)
    y = np.eye(n_classes)[y_indices]  # One-hot encoding
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    
    # Create and train network
    network = create_simple_classifier(n_features, 20, n_classes)
    
    print("Training network...")
    history = network.train(X_train, y_train, epochs=50, learning_rate=0.01, 
                           validation_data=(X_test, y_test), verbose=False)
    
    # Evaluate
    train_loss, train_acc = network.evaluate(X_train, y_train)
    test_loss, test_acc = network.evaluate(X_test, y_test)
    
    print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    print(f"Testing - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
    
    # Test individual components
    print("\nTesting individual components...")
    
    # Test activation functions
    x = np.array([-2, -1, 0, 1, 2])
    print(f"ReLU({x}) = {ActivationFunction.relu(x)}")
    print(f"Sigmoid({x}) = {ActivationFunction.sigmoid(x)}")
    
    # Test layers
    dense_layer = DenseLayer(5, 3)
    test_input = np.random.randn(2, 5)
    output = dense_layer.forward(test_input)
    print(f"Dense layer output shape: {output.shape}")
    
    print("Neural Network test completed successfully!")
    
    return network


if __name__ == "__main__":
    test_neural_network()

