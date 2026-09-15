import math
import numpy as np

def softmax(scores: list[float]) -> list[float]:
    scores = np.array(scores)
    den = np.sum(np.exp(scores-np.max(scores)))
    return np.exp(scores-np.max(scores))/den