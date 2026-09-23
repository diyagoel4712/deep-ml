import numpy as np

def softmax(scores: list[float]) -> list[float]:
    scores = np.array(scores)
    # subtracting the max to prevent overflow  
    scores_diff = scores-np.max(scores)
    den = np.sum(np.exp(scores_diff)) 
    return np.exp(scores_diff)/den