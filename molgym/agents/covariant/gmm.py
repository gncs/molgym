from abc import ABC

import torch
import torch.distributions as D
from torch.distributions import MixtureSameFamily


class GaussianMixtureModel(MixtureSameFamily, ABC):
    def __init__(
        self,
        log_probs: torch.Tensor,
        means: torch.Tensor,
        stds: torch.Tensor,
        validate_args=None,
    ) -> None:
        categoricals = D.Categorical(logits=log_probs, validate_args=validate_args)
        normals = D.Normal(loc=means, scale=stds, validate_args=validate_args)
        super().__init__(mixture_distribution=categoricals, component_distribution=normals, validate_args=validate_args)

    def argmax(self, count=128) -> torch.Tensor:
        # This can also be implemented using the EM algorithm
        # http://www.cs.columbia.edu/~jebara/htmlpapers/ARL/node61.html
        samples = self.sample(torch.Size((count, )))  # (samples, batches)
        log_probs = self.log_prob(samples)  # (samples, batches)
        indices = torch.argmax(log_probs, dim=0).unsqueeze(0)  # (1, batches)
        result = torch.gather(samples, dim=0, index=indices)  # (1, batches)
        return result.squeeze(0)  # (batches, )
