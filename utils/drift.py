import numpy as np

def detect_drift(reference_data, new_data, threshold=0.1):
    """
    Simple drift detection using mean difference
    """
    ref_mean = np.mean(reference_data)
    new_mean = np.mean(new_data)

    drift_score = abs(ref_mean - new_mean) / (abs(ref_mean) + 1e-8)

    drift_detected = drift_score > threshold

    return {
        "drift_score": float(drift_score),
        "drift_detected": drift_detected
    }