import numpy as np


def calibrate_probability(prob: float, temperature: float = 1.0, eps: float = 1e-6) -> float:
    """Apply temperature scaling to probability in a numerically stable way."""
    p = float(np.clip(prob, eps, 1.0 - eps))
    t = max(float(temperature), eps)
    logit = np.log(p / (1.0 - p))
    calibrated = 1.0 / (1.0 + np.exp(-logit / t))
    return float(np.clip(calibrated, 0.0, 1.0))
