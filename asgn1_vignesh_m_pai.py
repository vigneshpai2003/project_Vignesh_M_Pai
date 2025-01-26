import numpy as np


class LinReg:
    def __init__(self, data):
        self.data = data
        self.data_size = len(data)
        x, y = zip(*data)
        self.x_data = np.array(x)
        self.y_data = np.array(y)

        if len(self.x_data.shape) == 1:
            self.x_data = self.x_data.reshape(-1, 1)
            self.x_shape = 1
        else:
            self.x_shape = self.x_data.shape[1]

        if len(self.y_data.shape) == 1:
            self.y_data = self.y_data.reshape(-1, 1)
            self.y_shape = 1
        else:
            self.y_shape = self.y_data.shape[1]

        self.w = np.zeros((self.y_shape, self.x_shape))

    @property
    def is_scalar(self) -> bool:
        return self.x_shape == 1 and self.y_shape == 1

    @staticmethod
    def arize(arr):
        if hasattr(arr, '__iter__'):
            return np.array(arr)
        else:
            return np.array([arr])

    def get_weights(self) -> np.ndarray:
        if self.is_scalar:
            return self.w.flatten()
        return self.w

    def forward(self, x):
        x = __class__.arize(x)

        if self.is_scalar:
            return self.get_weights()[0] * x[0]

        return self.w @ np.array(x)

    @staticmethod
    def loss(y, y_pred) -> float:
        y = __class__.arize(y)
        y_pred = __class__.arize(y_pred)

        return np.mean((y - y_pred) ** 2)

    @staticmethod
    def gradient(x, y, y_pred):
        x = __class__.arize(x)
        y = __class__.arize(y)
        y_pred = __class__.arize(y_pred)

        if x.shape[0] == 1 and y.shape[0] == 1:
            return 2 * (y_pred[0] - y[0]) * x[0]

        return 2 * np.outer(y_pred - y, x) / y.shape[0]

    def fit(self, learning_rate: float = 0.01, n_iters: int = 200) -> None:
        for _ in range(n_iters):
            for i in range(self.data_size):
                x = self.x_data[i]
                y = self.y_data[i]
                y_pred = self.forward(x)
                grad = self.gradient(x, y, y_pred)
                self.w -= learning_rate * grad

