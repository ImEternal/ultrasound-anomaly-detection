import numpy as np


def generate_signal(n=200, defect=False):
    t = np.linspace(0, 10, n)

    signal = np.sin(t) + 0.1 * np.random.randn(n)

    if defect:
        # künstlicher Fehler (Spike)
        idx = np.random.randint(20, n - 20)
        signal[idx:idx + 5] += np.random.uniform(2, 4)

    return t, signal


def create_dataset(samples=1000):
    X = []
    y = []

    for _ in range(samples):
        defect = np.random.rand() > 0.5
        _, sig = generate_signal(defect=defect)

        X.append(sig)
        y.append(int(defect))

    return np.array(X), np.array(y)


if __name__ == '__main__':
    print(create_dataset())
