import math
import numpy as np

def sigmoid(prob):
	return 1/(1+math.exp(-prob))

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	features = np.array(features)
	weights = np.array(weights)
	labels = np.array(labels)
	logits = [np.dot(feature, weights) + bias for feature in features]
	probabilities = np.array([round(sigmoid(logit),4) for logit in logits])
	mse = round(np.mean(np.square(probabilities-labels)),4)
	return probabilities, mse