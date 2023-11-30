import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_hidden = np.random.randn(1, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_output = np.random.randn(1, self.output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_propagation(self, inputs):
        # Input to hidden layer
        self.hidden_output = self.sigmoid(np.dot(inputs, self.weights_input_hidden) + self.bias_hidden)

        # Hidden to output layer
        self.predicted_output = self.sigmoid(np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output)

        return self.predicted_output

    def backward_propagation(self, inputs, targets, learning_rate):
        # Calculate output layer error and delta
        output_error = targets - self.predicted_output
        output_delta = output_error * self.sigmoid_derivative(self.predicted_output)

        # Calculate hidden layer error and delta
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta) * learning_rate
        self.weights_input_hidden += inputs.T.dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta) * learning_rate

    def train(self, inputs, targets, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward_propagation(inputs)
            self.backward_propagation(inputs, targets, learning_rate)

            if epoch % 1000 == 0:
                loss = np.mean(np.square(targets - output))
                print(f"Epoch {epoch}: Loss = {loss}")

# Example usage:
# Create a neural network with 2 input neurons, 3 hidden neurons, and 1 output neuron
input_size = 2
hidden_size = 3
output_size = 1

# Input data (example XOR dataset)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Corresponding targets
y = np.array([[0], [1], [1], [0]])

# Initialize and train the neural network
nn = NeuralNetwork(input_size, hidden_size, output_size)
nn.train(X, y, epochs=10000, learning_rate=0.1)

# Test the trained network on new data
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predicted_output = nn.forward_propagation(test_data)
print("Predicted Output:")
print(predicted_output)
