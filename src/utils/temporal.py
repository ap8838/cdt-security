import numpy as np


<<<<<<< Updated upstream
def make_sliding_windows(x: np.ndarray, y=None, window: int = 5):
    """
    Convert feature matrix x into sliding windows.
    x shape: (N, D)
    output shape: (N-window+1, window*D)

    If y is provided, aligns labels to the last timestep of each window.
    """
    if window <= 1:
        return x, y

    n, d = x.shape
    if n < window:
        raise ValueError("Not enough samples for the given window size")

    Xw = []
    Yw = []

    for i in range(window - 1, n):
        Xw.append(x[i - window + 1 : i + 1].reshape(-1))
        if y is not None:
            Yw.append(y[i])

    Xw = np.asarray(Xw, dtype=np.float32)
    Yw = np.asarray(Yw, dtype=np.int64) if y is not None else None

    return Xw, Yw
=======
def make_sliding_windows(x, y=None, window=5):
    """
    Convert (N, F) -> (N - window + 1, F * window)
    Label is 1 if ANY anomaly appears inside the window.
    """
    xs, ys = [], []

    for i in range(len(x) - window + 1):
        xs.append(x[i : i + window].reshape(-1))
        if y is not None:
            ys.append(int(y[i : i + window].max()))

    xs = np.asarray(xs)
    ys = np.asarray(ys) if y is not None else None
    return xs, ys
>>>>>>> Stashed changes
