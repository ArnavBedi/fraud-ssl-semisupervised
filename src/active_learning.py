import numpy as np

def uncertainty_mask(proba, low=0.45, high=0.55):
    """
    proba: array of shape (n_samples, 2) with columns [p0, p1]
    Returns a boolean mask where the positive-class prob is in [low, high].
    """
    p1 = proba[:, 1]
    return (p1 >= low) & (p1 <= high)

def select_queries(X_pool, proba, budget=200, low=0.45, high=0.55, rng=None):
    """
    Pick up to `budget` points whose positive probability is near 0.5.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    mask = uncertainty_mask(proba, low, high)
    idx = np.where(mask)[0]
    if len(idx) > budget:
        idx = rng.choice(idx, size=budget, replace=False)
    return idx
