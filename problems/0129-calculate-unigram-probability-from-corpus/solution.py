def unigram_probability(corpus: str, word: str) -> float:
    probs = {}
    for token in corpus.split():
        probs[token] = probs.get(token,0) + 1
    den = sum(probs.values())
    return probs[word]/den