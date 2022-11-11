from torch import Tensor

class Normalizer():
    def __init__(self, x: Tensor) -> None:
        self.u = x.mean(dim=0, keepdim=True)
        self.s = x.std(dim=0, unbiased=False, keepdim=True)

    def normalize(self, x: Tensor) -> Tensor:
        x = (x-self.u)/self.s
        x[(self.s == 0).tile((len(x), 1))] = 0.
        return x

    def normalize_nonzero(self, x: Tensor) -> Tensor:
        return (x-self.u)/self.s

    @staticmethod
    def normalize_nonzero_(x: Tensor) -> Tensor:
        u = x.mean(dim=0, keepdim=True)
        s = x.std(dim=0, unbiased=False, keepdim=True)
        return (x-u)/s
