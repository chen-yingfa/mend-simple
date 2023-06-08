
import torch
from torch import Tensor
import torch.nn as nn
from typing import Tuple

from . import nn as local_nn


def update_counter(x: Tensor, m: Tensor, s, k) -> Tuple[Tensor, Tensor]:
    """
    Calculates the updated mean and variance given a new observation and
    the previous mean, variance, and number of observations.

    Args:
        x (float): The new observation.
        m (float): The previous mean.
        s (float): The previous variance.
        k (int): The number of observations.

    Returns:
        Tuple[float, float]: The updated mean and variance.
    """
    new_m = m + (x - m) / k
    new_s = s + (x - m) * (x - new_m)
    return new_m, new_s


class GradientTransform(nn.Module):
    def __init__(
        self,
        x_dim: int,
        delta_dim: int,
        num_modes: int,
        combine: bool = True,
        x_only: bool = False,
        delta_only: bool = False,
        one_sided: bool = False,
        n_hidden: int = 1,
        hidden_dim: int = -1,
        mlp_class: str = "IDMLP",
        init: str = "id",
        act: str = "relu",
        rank: int = 1920,
        norm: bool = True,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.delta_dim = delta_dim
        self.combine = combine
        self.x_only = x_only
        self.delta_only = delta_only
        self.one_sided = one_sided
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.mlp_class = mlp_class
        self.init = init
        self.act = act
        self.rank = rank
        self.norm = norm

        if combine and (one_sided or x_only or delta_only):
            raise ValueError("combine cannot be used with one-sided GTN variants")

        self.norm_init = False
        self.register_buffer("u_mean", torch.full((x_dim,), float("nan")))
        self.register_buffer("v_mean", torch.full((delta_dim,), float("nan")))
        self.register_buffer("u_std", torch.full((x_dim,), float("nan")))
        self.register_buffer("v_std", torch.full((delta_dim,), float("nan")))
        self.register_buffer("u_s", torch.full((x_dim,), float("nan")))
        self.register_buffer("v_s", torch.full((delta_dim,), float("nan")))
        self.register_buffer("k", torch.full((1,), float("nan")))

        MlpClass = getattr(local_nn, mlp_class)
        print(f"Building Gradient Transform with MLP class {MlpClass}")

        def delta_net():
            return MlpClass(
                delta_dim,
                delta_dim,
                delta_dim * 2,
                n_hidden,
                init=init,
                act=act,
                rank=rank,
                n_modes=num_modes,
            )

        def x_net():
            return MlpClass(
                x_dim,
                x_dim,
                x_dim * 2,
                n_hidden,
                init=init,
                act=act,
                rank=rank,
                n_modes=num_modes,
            )

        def combined_net():
            return MlpClass(
                delta_dim + x_dim,
                delta_dim + x_dim,
                (delta_dim + x_dim) * 2,
                n_hidden,
                init=init,
                act=act,
                rank=rank,
                n_modes=num_modes,
            )

        def ID():
            return lambda x, mode=None: x

        if combine:
            self.mlp = combined_net()
        elif one_sided:
            if x_dim > delta_dim:
                self.mlp1, self.mlp2 = ID(), delta_net()
            else:
                self.mlp1, self.mlp2 = x_net(), ID()
        elif x_only:
            self.mlp1, self.mlp2 = x_net(), ID()
        elif delta_only:
            self.mlp1, self.mlp2 = ID(), delta_net()
        else:
            self.mlp1, self.mlp2 = x_net(), delta_net()

    def forward(self, u, v, param_idx=None):
        u, v = u.to(torch.float32), v.to(torch.float32)

        u_ = u.view(-1, u.shape[-1])
        v_ = v.view(-1, v.shape[-1])

        nz_mask = (u_ != 0).any(-1) * (v_ != 0).any(
            -1
        )  # Skip batch elements with zero grad
        u_ = u_[nz_mask]
        v_ = v_[nz_mask]

        if self.training:
            for idx in range(u_.shape[0]):
                if not self.norm_init:
                    self.u_mean = u_[idx].clone().detach()
                    self.v_mean = v_[idx].clone().detach()
                    self.u_s.zero_()
                    self.v_s.zero_()
                    self.k[:] = 1
                    self.norm_init = True
                else:
                    self.k += 1
                    self.u_mean, self.u_s = update_counter(
                        u_[idx], self.u_mean, self.u_s, self.k
                    )
                    self.v_mean, self.v_s = update_counter(
                        v_[idx], self.v_mean, self.v_s, self.k
                    )

            if self.k < 2:
                raise RuntimeError(
                    f"Can't perform normalization with only {self.k} samples so far"
                )
            self.u_std = (self.u_s / (self.k - 1)) ** 0.5
            self.v_std = (self.v_s / (self.k - 1)) ** 0.5

        if self.norm:
            u_input = (u_ - self.u_mean) / (self.u_std + 1e-7)
            v_input = (v_ - self.v_mean) / (self.v_std + 1e-7)
        else:
            u_input = u_
            v_input = v_

        if self.combine:
            output = self.mlp(torch.cat((u_input, v_input), -1), mode=param_idx)
            out1, out2 = output.split([u.shape[-1], v.shape[-1]], -1)
            return out1, out2
        else:
            return self.mlp1(u_input, mode=param_idx), self.mlp2(
                v_input, mode=param_idx
            )
