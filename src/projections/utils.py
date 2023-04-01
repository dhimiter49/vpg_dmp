import numpy as np
import torch as ch
from typing import Tuple, Union, Iterable, Any

from nets.rl_nets import GaussianPolicy


def mean_distance(policy, mean, mean_other, std_other=None, scale_prec=False):
    """
    Compute mahalanobis distance for mean or euclidean distance
    Args:
        policy: policy instance
        mean: current mean vectors
        mean_other: old mean vectors
        std_other: scaling covariance matrix
        scale_prec: True computes the mahalanobis distance based on std_other for scaling. False the Euclidean distance.

    Returns:
        Mahalanobis distance or Euclidean distance between mean vectors
    """

    if scale_prec:
        # maha objective for mean
        mean_part = policy.maha(mean, mean_other, std_other)
    else:
        # euclidean distance for mean
        # mean_part = ch.norm(mean_other - mean, ord=2, axis=1) ** 2
        mean_part = ((mean_other - mean) ** 2).sum(1)

    return mean_part


def gaussian_kl(policy: GaussianPolicy, p: Tuple[ch.Tensor, ch.Tensor],
                q: Tuple[ch.Tensor, ch.Tensor]) -> Tuple[ch.Tensor, ch.Tensor]:
    """
    Get the expected KL divergence between two sets of Gaussians over states -
    Calculates E KL(p||q): E[sum p(x) log(p(x)/q(x))] in closed form for Gaussians.

    Args:
        policy: policy instance
        p: first distribution tuple (mean, var)
        q: second distribution tuple (mean, var)

    Returns:

    """

    mean, std = p
    mean_other, std_other = q
    k = mean.shape[-1]

    maha_part = .5 * policy.maha(mean, mean_other, std_other)

    det_term = policy.log_determinant(std)
    det_term_other = policy.log_determinant(std_other)

    # cov = policy.covariance(std)
    # prec_other = policy.precision(std_other)
    # trace_part = torch_batched_trace(prec_other @ cov)
    trace_part = torch_batched_trace_square(ch.linalg.solve_triangular(std_other, std, upper=False))
    cov_part = .5 * (trace_part - k + det_term_other - det_term)

    return maha_part, cov_part


def gaussian_frobenius(policy: GaussianPolicy, p: Tuple[ch.Tensor, ch.Tensor], q: Tuple[ch.Tensor, ch.Tensor],
                       scale_prec: bool = False, return_cov: bool = False) \
        -> Union[Tuple[ch.Tensor, ch.Tensor], Tuple[ch.Tensor, ch.Tensor, ch.Tensor, ch.Tensor]]:
    """
    Compute (p - q_values) (L_oL_o^T)^-1 (p - 1)^T + |LL^T - L_oL_o^T|_F^2 with p,q_values ~ N(y, LL^T)
    Args:
        policy: current policy
        p: mean and chol of gaussian p
        q: mean and chol of gaussian q_values
        return_cov: return cov matrices for further computations
        scale_prec: scale objective with precision matrix

    Returns: mahalanobis distance, squared frobenius norm

    """
    mean, chol = p
    mean_other, chol_other = q

    mean_part = mean_distance(policy, mean, mean_other, chol_other, scale_prec)

    # frob objective for cov
    cov_other = policy.covariance(chol_other)
    cov = policy.covariance(chol)
    diff = cov_other - cov
    # Matrix is real symmetric PSD, therefore |A @ A^H|^2_F = tr{A @ A^H} = tr{A @ A}
    cov_part = torch_batched_trace(diff @ diff)

    if return_cov:
        return mean_part, cov_part, cov, cov_other

    return mean_part, cov_part


def gaussian_wasserstein_commutative(policy: GaussianPolicy, p: Tuple[ch.Tensor, ch.Tensor],
                                     q: Tuple[ch.Tensor, ch.Tensor], scale_prec=False) -> Tuple[ch.Tensor, ch.Tensor]:
    """
    Compute mean part and cov part of W_2(p || q_values) with p,q_values ~ N(y, SS).
    This version DOES assume commutativity of both distributions, i.e. covariance matrices.
    This is less general and assumes both distributions are somewhat close together.
    When scale_prec is true scale both distributions with old precision matrix.
    Args:
        policy: current policy
        p: mean and sqrt of gaussian p
        q: mean and sqrt of gaussian q_values
        scale_prec: scale objective by old precision matrix.
                    This penalizes directions based on old uncertainty/covariance.

    Returns: mean part of W2, cov part of W2

    """
    mean, sqrt = p
    mean_other, sqrt_other = q

    mean_part = mean_distance(policy, mean, mean_other, sqrt_other, scale_prec)

    cov = policy.covariance(sqrt)
    if scale_prec:
        # cov constraint scaled with precision of old dist
        batch_dim, dim = mean.shape

        identity = ch.eye(dim, dtype=sqrt.dtype, device=sqrt.device)
        sqrt_inv_other = ch.linalg.solve(sqrt_other, identity)
        c = sqrt_inv_other @ cov @ sqrt_inv_other

        cov_part = torch_batched_trace(identity + c - 2 * sqrt_inv_other @ sqrt)

    else:
        # W2 objective for cov assuming normal W2 objective for mean
        cov_other = policy.covariance(sqrt_other)
        cov_part = torch_batched_trace(cov_other + cov - 2 * sqrt_other @ sqrt)

    return mean_part, cov_part


def gaussian_wasserstein_non_commutative(policy: GaussianPolicy, p: Tuple[ch.Tensor, ch.Tensor],
                                         q: Tuple[ch.Tensor, ch.Tensor], scale_prec=False,
                                         return_eig=False) -> Union[Tuple[ch.Tensor, ch.Tensor],
                                                                    Tuple[ch.Tensor, ch.Tensor, ch.Tensor, ch.Tensor]]:
    """
    Compute mean part and cov part of W_2(p || q_values) with p,q_values ~ N(y, SS)
    This version DOES NOT assume commutativity of both distributions, i.e. covariance matrices.
    This is more general an does not make any assumptions.
    When scale_prec is true scale both distributions with old precision matrix.
    Args:
        policy: current policy
        p: mean and sqrt of gaussian p
        q: mean and sqrt of gaussian q_values
        scale_prec: scale objective by old precision matrix.
                    This penalizes directions based on old uncertainty/covariance.
        return_eig: return eigen decomp for further computation

    Returns: mean part of W2, cov part of W2

    """
    mean, sqrt = p
    mean_other, sqrt_other = q
    batch_dim, dim = mean.shape

    mean_part = mean_distance(policy, mean, mean_other, sqrt_other, scale_prec)

    cov = policy.covariance(sqrt)

    if scale_prec:
        # cov constraint scaled with precision of old dist
        # W2 objective for cov assuming normal W2 objective for mean
        identity = ch.eye(dim, dtype=sqrt.dtype, device=sqrt.device)
        sqrt_inv_other = ch.linalg.solve(sqrt_other, identity)
        c = sqrt_inv_other @ cov @ sqrt_inv_other

        # compute inner parenthesis of trace in W2,
        # Only consider lower triangular parts, given cov/sqrt(cov) is symmetric PSD.
        eigvals, eigvecs = ch.symeig(c, eigenvectors=return_eig, upper=False)
        # make use of the following property to compute the trace of the root: 𝐴^2𝑥=𝐴(𝐴𝑥)=𝐴𝜆𝑥=𝜆(𝐴𝑥)=𝜆^2𝑥
        cov_part = torch_batched_trace(identity + c) - 2 * eigvals.sqrt().sum(1)

    else:
        # W2 objective for cov assuming normal W2 objective for mean
        cov_other = policy.covariance(sqrt_other)

        # compute inner parenthesis of trace in W2,
        # Only consider lower triangular parts, given cov/sqrt(cov) is symmetric PSD.
        eigvals, eigvecs = ch.symeig(cov @ cov_other, eigenvectors=return_eig, upper=False)
        # make use of the following property to compute the trace of the root: 𝐴^2𝑥=𝐴(𝐴𝑥)=𝐴𝜆𝑥=𝜆(𝐴𝑥)=𝜆^2𝑥
        cov_part = torch_batched_trace(cov_other + cov) - 2 * eigvals.sqrt().sum(1)

    if return_eig:
        return mean_part, cov_part, eigvals, eigvecs

    return mean_part, cov_part


def constraint_values(proj_type, policy: GaussianPolicy, p: Tuple[ch.Tensor, ch.Tensor],
                      q: Tuple[ch.Tensor, ch.Tensor], scale_prec: bool = True):
    """
    Computes the relevant metrics for a given batch of predictions.
    Args:
        proj_type: type of projection to compute the metrics for
        policy: current policy
        p: mean and std of gaussian p
        q: mean and std of gaussian q_values
        scale_prec: for W2 projection, use version scaled with precision matrix

    Returns: entropy, mean_part, cov_part, kl

    """
    if proj_type == "w2":
        mean_part, cov_part = gaussian_wasserstein_commutative(policy, p, q, scale_prec=scale_prec)

    elif proj_type == "w2_non_com":
        # For this case only the sum is relevant, no individual projections for mean and std make sense
        mean_part, cov_part = gaussian_wasserstein_non_commutative(policy, p, q, scale_prec=scale_prec)

    elif proj_type == "frob":
        mean_part, cov_part = gaussian_frobenius(policy, p, q, scale_prec=scale_prec)

    else:
        # we assume kl projection as default (this is also true for PPO)
        mean_part, cov_part = gaussian_kl(policy, p, q)

    entropy = policy.entropy(p)
    mean_kl, cov_kl = gaussian_kl(policy, p, q)
    kl = mean_kl + cov_kl

    return entropy, mean_part, cov_part, kl


def get_entropy_schedule(schedule_type, total_train_steps, dim):
    """
    return entropy schedule callable with interface f(old_entropy, initial_entropy_bound, train_step)
    Args:
        schedule_type: which type of entropy schedule to use, one of [None, 'linear', or 'exp'].
        total_train_steps: total number of training steps to compute appropriate decay over time.
        dim: number of action dimensions to scale exp decay correctly.

    Returns:
        f(initial_entropy, target_entropy, temperature, step)
    """
    if schedule_type == "linear":
        return lambda initial_entropy, target_entropy, temperature, step: step * (
                target_entropy - initial_entropy) / total_train_steps + initial_entropy
    # if schedule_type == "linear_v2":
    #     return lambda old_entropy, beta, step: old_entropy - beta
    elif schedule_type == "exp":
        return lambda initial_entropy, target_entropy, temperature, step: dim * target_entropy + (
                initial_entropy - dim * target_entropy) * temperature ** (10 * step / total_train_steps)
    # elif schedule_type == "exp_v2":
    #     return lambda old_entropy, beta, step: old_entropy - (
    #             - step / (total_train_step / temperature) + beta.log()).exp()
    else:
        return lambda initial_entropy, target_entropy, temperature, step: initial_entropy.new([-np.inf])


def generate_minibatches(n, n_minibatches):
    """
    Generate n_minibatches sets of indices for N data points.
    Args:
        n: total number of data points
        n_minibatches: how many minibatches to generate

    Returns:
        np.ndarray of minibatched indices
    """
    state_indices = np.arange(n)
    np.random.shuffle(state_indices)
    return np.array_split(state_indices, n_minibatches)


def select_batch(index, *args) -> list:
    """
    For each argument select the value at index.
    Args:
        index: index of values to select
        *args: data

    Returns:
        list of indexed value
    """
    return [v[index] for v in args]


def tensorize(x, cpu=True, dtype=ch.float32):
    return cpu_tensorize(x, dtype) if cpu else gpu_tensorize(x, dtype)


def gpu_tensorize(x, dtype=None):
    """
    Utility function for turning arrays into cuda tensors
    Args:
        x: data
        dtype: dtype to generate

    Returns:
        gpu tensor version of x
    """
    return cpu_tensorize(x, dtype).cuda()


def cpu_tensorize(x, dtype=None):
    """
    Utility function for turning arrays into cpu tensors
    Args:
        x: data
        dtype: dtype to generate

    Returns:
        cpu tensor version of x
    """
    dtype = dtype if dtype else x.dtype
    if not isinstance(x, ch.Tensor):
        x = ch.tensor(x)
    return x.type(dtype)


def torch_batched_trace(x) -> ch.Tensor:
    """
    Compute trace in n,m of batched matrix
    Args:
        x: matrix with shape [a,...l, n, m]

    Returns: trace with shape [a,...l]

    """
    return ch.diagonal(x, dim1=-2, dim2=-1).sum(-1)


def torch_batched_trace_square(x):
    """
    Compute trace in n,m of squared batched matrix XX^{T}
    Args:
        x: matrix with shape [a,..., l, n, m]

    Returns: trace with shape [a,...l]

    """
    n = x.size(-1)
    m = x.size(-2)
    flat_trace = x.reshape(-1, m * n).square().sum(-1)
    return flat_trace.reshape(x.shape[:-2])


def get_optimizer(optimizer_type: str, model_parameters: Union[Iterable[ch.Tensor], Iterable[dict]],
                  learning_rate: float, **kwargs):
    """
    Get optimizer instance for given model parameters
    Args:
        model_parameters:
        optimizer_type:
        learning_rate:
        **kwargs:

    Returns:

    """
    if optimizer_type.lower() == "sgd":
        return optim.SGD(model_parameters, learning_rate, **kwargs)
    elif optimizer_type.lower() == "sgd_momentum":
        momentum = kwargs.pop("momentum") if kwargs.get("momentum") else 0.9
        return optim.SGD(model_parameters, learning_rate, momentum=momentum, **kwargs)
    elif optimizer_type.lower() == "adam":
        return optim.Adam(model_parameters, learning_rate, **kwargs)
    elif optimizer_type.lower() == "adamw":
        return optim.AdamW(model_parameters, learning_rate, **kwargs)
    elif optimizer_type.lower() == "adagrad":
        return optim.adagrad.Adagrad(model_parameters, learning_rate, **kwargs)
    else:
        ValueError(f"Optimizer {optimizer_type} is not supported.")


def get_numpy(x):
    """
    Convert torch tensor to numpy
    Args:
        x: torch.Tensor

    Returns:
        numpy tensor version of x

    """
    return x.cpu().detach().numpy()


def sqrtm_newton(x: ch.Tensor, **kwargs: Any):
    """
    From: https://github.com/msubhransu/matrix-sqrt/blob/master/matrix_sqrt.py
    License: MIT

    Compute the Sqrt of a matrix based on Newton-Schulz algorithm
    """
    num_iters = kwargs.get("num_iters") or 10

    batch_size = x.shape[0]
    dim = x.shape[-1]
    dtype = x.dtype

    normA = x.pow(2).sum(dim=1).sum(dim=1).sqrt()
    Y = x / normA.view(batch_size, 1, 1).expand_as(x)
    I = 3.0 * ch.eye(dim, dtype=dtype)
    Z = ch.eye(dim, dtype=dtype)
    for i in range(num_iters):
        T = 0.5 * (I - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    sA = Y * normA.sqrt().view(batch_size, 1, 1).expand_as(x)
    return sA
