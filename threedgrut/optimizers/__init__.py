# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# Selective Adam implementation was adpoted from gSplat library (https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/optimizers/selective_adam.py),
# which is based on the original implementation https://github.com/humansensinglab/taming-3dgs that uderlines the work
#
# Taming 3DGS: High-Quality Radiance Fields with Limited Resources by
# Saswat Subhajyoti Mallick*, Rahul Goel*, Bernhard Kerbl, Francisco Vicente Carrasco, Markus Steinberger and Fernando De La Torre
#
# If you use this code in your research, please cite the above works.


import torch
import math


_optimizer_plugin = None


def load_optimizer_plugin():
    global _optimizer_plugin
    if _optimizer_plugin is None:
        try:
            from . import lib_optimizers_cc as optimizers_cc
        except ImportError:
            from .setup_optimizers import setup_lib_optimizers_cc

            setup_lib_optimizers_cc()  # Setup the C++ extension for the optimizer plugin
            import lib_optimizers_cc as optimizers_cc

        _optimizer_plugin = optimizers_cc


class SelectiveAdam(torch.optim.Adam):
    """
    A custom optimizer that extends the standard Adam optimizer by
    incorporating selective updates.

    This class is useful for situations where only a subset of parameters
    should be updated at each step, such as in sparse models or in cases where
    parameter visibility is controlled by an external mask.

    Additionally, the operations are fused into a single kernel. This optimizer
    leverages the `adam` function from a CUDA backend for
    optimized sparse updates.

    This is one of the two optimizers mentioned in the Taming3DGS paper. This implementation
    also references gsplat's SelectiveAdam optimizer.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        eps (float): Term added to the denominator to improve numerical stability (default: 1e-8).
        betas (Tuple[float, float]): Coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999)).

    Examples:

        >>> N = 100
        >>> param = torch.randn(N, requires_grad=True)
        >>> optimizer = SelectiveAdam([param], eps=1e-8, betas=(0.9, 0.999))
        >>> visibility_mask = torch.cat([torch.ones(50), torch.zeros(50)])  # Visible first half, hidden second half

        >>> # Forward pass
        >>> loss = torch.sum(param ** 2)

        >>> # Backward pass
        >>> loss.backward()

        >>> # Optimization step with selective updates
        >>> optimizer.step(visibility=visibility_mask)

    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08):
        super().__init__(params=params, lr=lr, eps=eps, betas=betas)
        load_optimizer_plugin()

    @torch.no_grad()
    def step(self, visibility):

        for group in self.param_groups:

            lr = group["lr"]
            eps = group["eps"]
            beta1, beta2 = group["betas"]

            assert (
                len(group["params"]) == 1
            ), "More than one tensor in group is not supported"

            param = group["params"][0]
            if param.grad is None:
                continue

            # Lazy state initialization
            state = self.state[param]
            if len(state) == 0:
                state["step"] = torch.tensor(0.0, dtype=torch.float32)
                state["exp_avg"] = torch.zeros_like(
                    param, memory_format=torch.preserve_format
                )
                state["exp_avg_sq"] = torch.zeros_like(
                    param, memory_format=torch.preserve_format
                )

            stored_state = self.state.get(param, None)
            exp_avg = stored_state["exp_avg"]
            exp_avg_sq = stored_state["exp_avg_sq"]

            _optimizer_plugin.selective_adam_update(
                param.contiguous(),
                param.grad.contiguous(),
                exp_avg.contiguous(),
                exp_avg_sq.contiguous(),
                visibility.bool().squeeze(),
                lr,
                beta1,
                beta2,
                eps,
            )


class SGHMC(torch.optim.Optimizer):
    """Simplified SGHMC optimizer with optional Fisher preconditioning."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        damping: float = 1e-5,
        fisher_alpha: float = 0.95,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            damping=damping,
            fisher_alpha=fisher_alpha,
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, visibility=None, noise_scale: float = 1.0, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            damping = group["damping"]
            alpha = group["fisher_alpha"]

            noise_std = math.sqrt(2.0 * lr * (1 - momentum)) * noise_scale

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if visibility is not None:
                    view = visibility.view(-1, *([1] * (grad.dim() - 1)))
                    grad = grad * view
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state.setdefault(p, {})
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                if "fisher" not in state:
                    state["fisher"] = torch.zeros_like(p)

                state["fisher"].mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                precond = grad / (state["fisher"].sqrt() + damping)

                buf = state["momentum_buffer"]
                if visibility is not None:
                    view = visibility.view(-1, *([1] * (buf.dim() - 1)))
                    noise = torch.randn_like(p) * noise_std
                    buf.mul_(momentum).add_(precond * view).add_(noise * view)
                else:
                    buf.mul_(momentum).add_(precond).add_(torch.randn_like(p) * noise_std)

                p.add_(buf, alpha=-lr)

        return None

