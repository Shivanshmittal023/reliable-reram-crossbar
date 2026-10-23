"""SVD-factorised linear layer: W ~= U · diag(S) · Vh.

On the crossbar the layer becomes two arrays back to back:
    x --[ ΣVᵀ  (k x in) ]--> h --[ U (out x k) ]--> y
Component r is row r of ΣVᵀ together with column r of U. S is kept as its own
parameter so that the importance I_r = |dL/dσ_r| can be read straight from S.grad.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def param_preserving_rank(out_features: int, in_features: int) -> int:
    """k = floor(mn / (m + n)): the factorised layer has the same parameter count as W."""
    return (out_features * in_features) // (out_features + in_features)


class SVDLinear(nn.Module):
    def __init__(self, U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        self.in_features = Vh.shape[1]
        self.out_features = U.shape[0]
        self.rank = S.shape[0]
        self.U = nn.Parameter(U.contiguous())
        self.S = nn.Parameter(S.contiguous())
        self.Vh = nn.Parameter(Vh.contiguous())
        self.bias = nn.Parameter(bias.clone()) if bias is not None else None

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int) -> "SVDLinear":
        W = linear.weight.data.float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        k = min(rank, S.shape[0])
        bias = linear.bias.data.float() if linear.bias is not None else None
        return cls(U[:, :k], S[:k], Vh[:k], bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.linear(x, self.Vh) * self.S
        return F.linear(h, self.U, self.bias)

    @torch.no_grad()
    def crossbar_factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(U, ΣVᵀ): the two matrices that get programmed into crossbars."""
        return self.U.detach().clone(), (self.S[:, None] * self.Vh).detach().clone()

    @torch.no_grad()
    def effective_weight(self) -> torch.Tensor:
        U, SVh = self.crossbar_factors()
        return U @ SVh

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, rank={self.rank}, bias={self.bias is not None}"
