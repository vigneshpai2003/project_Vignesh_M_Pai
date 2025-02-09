import torch


class LinReg:
    def __init__(self, data):
        self.data = data
        self.data_size = len(data)
        self.w = torch.tensor([1], dtype=torch.float64, requires_grad=True)

    def get_weights(self) -> torch.tensor:
        return self.w

    def forward(self, x):
        return self.w * x

    def fit(self, learning_rate: float = 0.01, n_iters: int = 200) -> None:
        loss = torch.nn.MSELoss()
        optimizer = torch.optim.SGD([self.w], lr=learning_rate)
        for _ in range(n_iters):
            for (x_data, y_data) in self.data:
                x = torch.tensor([x_data], requires_grad=True,
                                 dtype=torch.float64)
                y = torch.tensor([y_data], dtype=torch.float64)
                y_pred = self.forward(x)

                optimizer.zero_grad()
                l = loss(y_pred, y)
                l.backward()
                optimizer.step()


data = [[1, 2], [2, 5], [3, 6], [4, 8], [5, 10]]
m = LinReg(data)
m.fit()

print(m.get_weights().item())
