import torch
import torch.distributions


def to_one_hot(indices: torch.Tensor, num_classes: int, device=torch.device('cpu')) -> torch.Tensor:
    """
    Generates one-hot encoding with <num_classes> classes from <indices>

    :param indices: (N x 1) tensor
    :param num_classes: number of classes
    :param device: torch device
    :return: (N x num_classes) tensor
    """
    shape = (*indices.shape[:-1], num_classes)
    oh = torch.zeros(shape, device=device).view(-1, num_classes)

    # scatter_ is the in-place version of scatter
    oh.scatter_(1, indices.view(-1, 1), 1)

    return oh.view(*shape)


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Computes the masked softmax from logits.

    :param logits: (N x ... x K) tensor
    :param mask: (N x ... x K) tensor
    :return: (N x ... x K) tensor
    """
    maxima, _ = torch.max(logits, dim=-1, keepdim=True)
    exps = torch.exp(logits - maxima)

    # Note: in-place operator must not be used here!
    exps = exps * mask
    return exps / torch.sum(exps, keepdim=True, dim=-1)


def init_layer(layer: torch.nn.Linear, w_scale=1.0) -> torch.nn.Linear:
    torch.nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)  # type: ignore
    torch.nn.init.constant_(layer.bias.data, 0)
    return layer


class MLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_units=(64, 64), gate=torch.nn.functional.relu):
        super().__init__()
        dims = (input_dim, ) + hidden_units
        self.layers = torch.nn.ModuleList(
            [init_layer(torch.nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.output_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers[:1]:
            x = self.gate(layer(x))
        x = self.layers[-1](x)
        return x
