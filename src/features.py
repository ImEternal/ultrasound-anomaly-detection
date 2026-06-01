import numpy as np

def extract_features(X):
    features = []

    for signal in X:
        f = [
            np.mean(signal),
            np.std(signal),
            np.max(signal),
            np.min(signal),
            np.ptp(signal)  # peak-to-peak
        ]
        features.append(f)

    return np.array(features)