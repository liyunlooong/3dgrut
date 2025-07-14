# SPDX-License-Identifier: Apache-2.0
# Utility functions for Unscented Transform operations
import torch
from typing import Callable, Tuple

def unscented_transform(
    mean: torch.Tensor,
    covariance: torch.Tensor,
    transform: Callable[[torch.Tensor], torch.Tensor],
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Unscented Transform to propagate mean and covariance.

    Args:
        mean: Tensor of shape (..., N).
        covariance: Tensor of shape (..., N, N).
        transform: Function applied to each sigma point.
        alpha: Spread of sigma points.
        beta: Beta parameter (typically 2).
        kappa: Secondary scaling parameter.

    Returns:
        Tuple of transformed mean and covariance.
    """
    n = mean.shape[-1]
    lambda_ = alpha ** 2 * (n + kappa) - n
    # Compute square root of covariance
    sqrt_cov = torch.linalg.cholesky((n + lambda_) * covariance)

    sigma_points = [mean]
    for i in range(n):
        offset = sqrt_cov[..., :, i]
        sigma_points.append(mean + offset)
        sigma_points.append(mean - offset)
    sigma_points = torch.stack(sigma_points, dim=0)

    w0_m = lambda_ / (n + lambda_)
    w0_c = w0_m + 1 - alpha ** 2 + beta
    wi = 1.0 / (2 * (n + lambda_))

    transformed = transform(sigma_points)
    mean_t = w0_m * transformed[0]
    for i in range(1, sigma_points.shape[0]):
        mean_t = mean_t + wi * transformed[i]

    diff0 = transformed[0] - mean_t
    cov_t = w0_c * (diff0[..., :, None] * diff0[..., None, :])
    for i in range(1, sigma_points.shape[0]):
        diff = transformed[i] - mean_t
        cov_t = cov_t + wi * (diff[..., :, None] * diff[..., None, :])
    return mean_t, cov_t
