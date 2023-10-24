"""Levenberg-Marquardt optimizer.

Notes:
    This is a reimplementation of the Levenberg-Marquardt optimizer from the fantastic
    ForceBalance [1] package. The original code is licensed under the BSD 3-clause
    license which can be found in the LICENSE_3RD_PARTY file.

References:
    [1]: https://github.com/leeping/forcebalance/blob/b395fd4b/src/optimizer.py
"""
import logging
import math
import typing

import pydantic
import torch

import smee.utils

_LOGGER = logging.getLogger(__name__)


ClosureFn = typing.Callable[
    [torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
]
CorrectFn = typing.Callable[[torch.Tensor], torch.Tensor]


class LevenbergMarquardtConfig(pydantic.BaseModel):
    """Configuration for the Levenberg-Marquardt optimizer."""

    type: typing.Literal["levenberg-marquardt"] = "levenberg-marquardt"

    trust_radius: float = pydantic.Field(
        0.2, description="Target trust radius.", gt=0.0
    )
    min_trust_radius: float = pydantic.Field(0.05, description="Minimum trust radius.")

    adaptive_factor: float = pydantic.Field(
        0.25, description="Adaptive trust radius adjustment factor.", gt=0.0
    )
    adaptive_damping: float = pydantic.Field(
        1.0, description="Adaptive trust radius adjustment damping.", gt=0.0
    )

    error_tolerance: float = pydantic.Field(
        1.0,
        description="Steps where the loss increases more than this amount are rejected.",
    )

    quality_threshold_low: float = pydantic.Field(
        0.25,
        description="The threshold below which the step is considered low quality.",
    )
    quality_threshold_high: float = pydantic.Field(
        0.75,
        description="The threshold above which the step is considered high quality.",
    )


def _invert_svd(matrix: torch.Tensor, threshold: float = 1e-12) -> torch.Tensor:
    """Invert a matrix using SVD.

    Args:
        matrix: The matrix to invert.
        threshold: The threshold below which singular values are considered zero.

    Returns:
        The inverted matrix.
    """
    u, s, vh = torch.linalg.svd(matrix)

    non_zero_idxs = s > threshold

    s_inverse = torch.zeros_like(s)
    s_inverse[non_zero_idxs] = 1.0 / s[non_zero_idxs]

    return vh.T @ torch.diag(s_inverse) @ u.T


def _solver(
    damping_factor: torch.Tensor, gradient: torch.Tensor, hessian: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve the Levenberg–Marquardt step.

    Args:
        damping_factor: The damping factor with ``shape=(1,)``.
        gradient: The gradient with ``shape=(n,)``.
        hessian: The Hessian with ``shape=(n, n)``.

    Returns:
        The step with ``shape=(n,)`` and the expected improvement with ``shape=()``.
    """

    hessian_regular = hessian + (damping_factor - 1) ** 2 * torch.eye(
        len(hessian), device=hessian.device, dtype=hessian.dtype
    )
    hessian_inverse = _invert_svd(hessian_regular)

    dx = -(hessian_inverse @ gradient)
    solution = 0.5 * dx @ hessian @ dx + (dx * gradient).sum()

    return dx, solution


def _damping_factor_loss_fn(
    damping_factor: torch.Tensor,
    gradient: torch.Tensor,
    hessian: torch.Tensor,
    trust_radius: float,
) -> torch.Tensor:
    """Computes the squared difference between the target trust radius and the step size
    proposed by the Levenberg–Marquardt solver.

    This is used when finding the optimal damping factor.

    Args:
        damping_factor: The damping factor with ``shape=(1,)``.
        gradient: The gradient with ``shape=(n,)``.
        hessian: The hessian with ``shape=(n, n)``.
        trust_radius: The target trust radius.

    Returns:
        The squared difference.
    """
    dx, _ = _solver(damping_factor, gradient, hessian)
    dx_norm = torch.linalg.norm(dx)

    _LOGGER.info(
        f"finding trust radius: length {dx_norm:.4e} (target {trust_radius:.4e})"
    )

    return (dx_norm - trust_radius) ** 2


def _step(
    gradient: torch.Tensor,
    hessian: torch.Tensor,
    trust_radius: torch.Tensor,
    initial_damping_factor: float = 1.0,
    min_eigenvalue: float = 1.0e-4,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Compute the Levenberg–Marquardt step.

    Args:
        gradient: The gradient with ``shape=(n,)``.
        hessian: The hessian with ``shape=(n, n)``.
        trust_radius: The target trust radius.
        initial_damping_factor: An initial guess of the Levenberg-Marquardt damping
            factor
        min_eigenvalue: Lower bound on hessian eigenvalue. If the smallest eigenvalue
            is smaller than this, a small amount of steepest descent is mixed in to
            try and correct this.

    Notes:
        * the code to 'excise' certain parameters is for now removed until its clear
          it is needed.
        * only trust region is implemented (i.e., only trust0 > 0 is supported)

    Returns:
        The step with ``shape=(n,)``, the expected improvement with ``shape=()``, and
        a boolean indicating whether the damping factor was adjusted.
    """
    from scipy import optimize

    eigenvalues, _ = torch.linalg.eigh(hessian)
    eigenvalue_smallest = eigenvalues.min()

    if eigenvalue_smallest < min_eigenvalue:
        # Mix in SD step if Hessian minimum eigenvalue is negative - experimental.
        adjacency = (
            max(min_eigenvalue, 0.01 * abs(eigenvalue_smallest)) - eigenvalue_smallest
        )

        _LOGGER.info(
            f"hessian has a small or negative eigenvalue ({eigenvalue_smallest:.1e}), "
            f"mixing in some steepest descent ({adjacency:.1e}) to correct this."
        )
        hessian += adjacency * torch.eye(
            hessian.shape[0], device=hessian.device, dtype=hessian.dtype
        )

    damping_factor = torch.tensor(1.0)

    dx, improvement = _solver(damping_factor, gradient, hessian)
    dx_norm = torch.linalg.norm(dx)

    adjust_damping = (dx_norm > trust_radius).item()

    if adjust_damping:
        # LPW tried a few optimizers and found Brent works well, but also that the
        # tolerance is fractional - if the optimized value is zero it takes a lot of
        # meaningless steps.
        damping_factor = optimize.brent(
            _damping_factor_loss_fn,
            (
                gradient.detach().cpu(),
                hessian.detach().cpu(),
                trust_radius.detach().cpu(),
            ),
            brack=(initial_damping_factor, initial_damping_factor * 4),
            tol=1e-6,
        )

        dx, improvement = _solver(damping_factor, gradient, hessian)
        dx_norm = torch.linalg.norm(dx)

    _LOGGER.info(f"trust-radius step found (length {dx_norm:.4e})")

    return dx, improvement, adjust_damping


def _reduce_trust_radius(
    dx_norm: torch.Tensor, config: LevenbergMarquardtConfig
) -> torch.Tensor:
    """Reduce the trust radius.

    Args:
        dx_norm: The size of the previous step.
        config: The optimizer config.

    Returns:
        The reduced trust radius.
    """
    trust_radius = max(
        dx_norm * (1.0 / (1.0 + config.adaptive_factor)), config.min_trust_radius
    )
    _LOGGER.info(f"reducing trust radius to {trust_radius:.4e}")

    return smee.utils.tensor_like(trust_radius, dx_norm)


def _update_trust_radius(
    dx_norm: torch.Tensor,
    step_quality: float,
    trust_radius: torch.Tensor,
    damping_adjusted: bool,
    config: LevenbergMarquardtConfig,
) -> torch.Tensor:
    """Adjust the trust radius based on the quality of the previous step.

    Args:
        dx_norm: The size of the previous step.
        step_quality: The quality of the previous step.
        trust_radius: The current trust radius.
        damping_adjusted: Whether the LM damping factor was adjusted during the
            previous step.
        config: The optimizer config.

    Returns:
        The updated trust radius.
    """

    if step_quality <= config.quality_threshold_low:
        trust_radius = max(
            dx_norm * (1.0 / (1.0 + config.adaptive_factor)),
            smee.utils.tensor_like(config.min_trust_radius, dx_norm),
        )
        _LOGGER.info(
            f"low quality step detected - reducing trust radius to {trust_radius:.4e}"
        )

    elif step_quality >= config.quality_threshold_high and damping_adjusted:
        trust_radius += (
            config.adaptive_factor
            * trust_radius
            * math.exp(
                -config.adaptive_damping * (trust_radius / config.trust_radius - 1.0)
            )
        )
        _LOGGER.info(f"updating trust radius to {trust_radius: .4e}")

    return trust_radius


class LevenbergMarquardt:
    """A Levenberg-Marquardt optimizer.

    Notes:
        This is a reimplementation of the Levenberg-Marquardt optimizer from the
        ForceBalance package, and so may differ from a standard implementation.
    """

    def __init__(self, config: LevenbergMarquardtConfig | None = None):
        self.config = config if config is not None else LevenbergMarquardtConfig()

        self._closure_prev = None
        self._trust_radius = torch.tensor(self.config.trust_radius)

    @torch.no_grad()
    def step(
        self,
        x: torch.Tensor,
        closure_fn: ClosureFn,
        correct_fn: CorrectFn | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs a single optimization step.

        Args:
            x: The initial guess of the parameters.
            closure_fn: The closure that computes the loss (``shape=()``), its
                gradient (``shape=(n,)``), and hessian (``shape=(n, n)``)..
            correct_fn: A function that can be used to correct the parameters after
                each step is taken and before the new loss is computed. This may
                include, for example, ensuring that vdW parameters are all positive.

        Returns:
            The optimized parameters.
        """

        correct_fn = correct_fn if correct_fn is not None else lambda x: x
        closure_fn = torch.enable_grad()(closure_fn)

        if self._closure_prev is None:
            # compute the initial loss, gradient and hessian
            self._closure_prev = closure_fn(x)

        if self._trust_radius.device != x.device:
            self._trust_radius = self._trust_radius.to(x.device)

        loss_prev, gradient_prev, hessian_prev = self._closure_prev

        dx, expected_improvement, damping_adjusted = _step(
            gradient_prev, hessian_prev, self._trust_radius
        )
        dx_norm = torch.linalg.norm(dx)

        x_next = correct_fn(x + dx).requires_grad_(x.requires_grad)

        loss_next, gradient_next, hessian_next = closure_fn(x_next)
        loss_delta = loss_next - loss_prev

        step_quality = loss_delta / expected_improvement

        if loss_next > (loss_prev + self.config.error_tolerance):
            # reject the 'bad' step and try again from where we were
            loss, gradient, hessian = (loss_prev, gradient_prev, hessian_prev)

            self._trust_radius = _reduce_trust_radius(dx_norm, self.config)
        else:
            # accept the step
            loss, gradient, hessian = (loss_next, gradient_next, hessian_next)
            x.data.copy_(x_next.data)

            self._trust_radius = _update_trust_radius(
                dx_norm, step_quality, self._trust_radius, damping_adjusted, self.config
            )

        self._closure_prev = (loss, gradient, hessian)
        return self._closure_prev