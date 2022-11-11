from collections.abc import Callable
from torch import nn, Tensor
from torch.nn import Module


class AutoEncoder(Module):

    __call__: Callable[..., Tensor]

    @staticmethod
    def nearest_2_pow(n: int):
        return 2**((n-1).bit_length())

    def __init__(self, in_features: int, device: str):
        super(AutoEncoder, self).__init__()
        self.widths = [*map(self.nearest_2_pow, (in_features//4, in_features//8))]
        self.encoder = nn.Sequential(
            nn.Linear(in_features, self.widths[0], device=device),
            nn.ReLU(),
            nn.Linear(self.widths[0], self.widths[1], device=device),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.widths[1], self.widths[0], device=device),
            nn.ReLU(),
            nn.Linear(self.widths[0], in_features, device=device),
        )

    def forward(self, input: Tensor):
        return self.decoder(self.encoder(input))

    def get_score(self, input: Tensor):
        return (input-self(input)).pow(2).mean(dim=1).sqrt()
