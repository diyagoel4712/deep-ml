import numpy as np

def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	epoch = 0
	weights = initial_weights
	bias = initial_bias
	mse_values = []
	for _ in range(epochs):
		# forward pass
		outputs = forward_pass(features, weights, bias)
		# compute loss
		loss = round(np.mean(np.square(outputs-labels)),4)
		mse_values.append(loss)
		# backpropagate the loss
		gradient = 2/len(labels)*(outputs-labels)*(outputs)*(1-outputs)
		# update the weights
		weights -= learning_rate*(np.matmul(features.T,gradient))
		bias -= learning_rate*(np.sum(gradient))

	return np.round(weights,4), np.round(bias,4), mse_values

def forward_pass(features, weights, bias):
	logits = np.matmul(features, weights) + bias
	probs = 1/(1+np.exp(-logits)) # sigmoid
	return probs