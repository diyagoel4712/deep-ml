import numpy as np
import math

def calculate_perplexity(probabilities: list[float]) -> float:
    """
    Calculate the perplexity of a language model given token probabilities.
    
    Args:
        probabilities: List of probabilities P(token_i | context) for each token
                      in the sequence, where each probability is in (0, 1]
    
    Returns:
        Perplexity value as a float
    """
    log_probabilities = np.log(np.array(probabilities))
    ppl = math.exp((-np.sum(log_probabilities))/len(probabilities))
    return ppl