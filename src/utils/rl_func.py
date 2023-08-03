import torch
from torch import nn
import numpy as np

INTRINSIC_REWARD_THRESHOLD = 160


def critic_decision(heatmap, mode="max"):
    if mode == "max":
        return int(torch.argmax(heatmap))
    elif mode == "softmax_categorical":
        return int(torch.distributions.categorical.Categorical(
            nn.functional.softmax(heatmap.flatten())
        ).sample())
    elif mode == "categorical":
        return int(torch.distributions.categorical.Categorical(
            heatmap.flatten()
        ).sample())


def pixel_pos_ret(returns):
    pixel_pos_penalty = returns[-1]["critic_penalty"]
    dist_to_tcp_ret = returns[-1]["critic_box_to_tcp_rew"]
    return dist_to_tcp_ret + pixel_pos_penalty


def intrinsic_push_rew(previous_obs, current_obs, function="step"):
    """
    Calculate an intrinsic reward for a pushing action based on the difference in obs
    between the state before and after executing the action.

    Args:
        previous_obs (torch.rensor): image tensor of previous obs
        current_obs (torch.rensor): image tensor of currnet_obs
    Return:
        (float): reward value
    """
    distance = torch.abs(previous_obs - current_obs)
    distance[distance > 0.3] = 0
    distance[distance < 0.01] = 0
    distance[distance > 0] = 1.0
    distance = torch.sum(distance).item()

    if function == "step":
        rew = 0.5 if distance > INTRINSIC_REWARD_THRESHOLD else 0.0
    if function == "linear":
        rew = 0.5 + (distance - INTRINSIC_REWARD_THRESHOLD) /\
              np.prod(current_obs.shape[-2:])

    return rew


def get_td_error(ret, pred_ret, gamma, terminal):
    """
    Calculate (TD) temporal difference error.

    Args:
        ret (torch.tensor): rewards
        pred_ret (torch.tensor): predicted reward (value function evaluation)
        gamm (float): dicount
        terminal (torch.tensor): (1 - done) flags related to the rewards
    """
    return ret[:-1] + gamma * terminal[:-1] * pred_ret[1:] - pred_ret[:-1]


def get_gae(ret, pred_ret, gamma, _lambda, dones):
    terminal = 1 - dones
    gae = get_td_error(ret, pred_ret, gamma, terminal)
    for i in reversed(range(len(gae) - 1)):
        gae[i] += gamma * _lambda * terminal[i + 1] * gae[i + 1]
    return gae.flatten()


def clip_IS_loss(logp, logp_old, error, clip):
    """
    Clip and normalize TD importance sampling error, based on PPO implementation.

    Args:
       logp (torch.tesor): log probabilities
       old_logp (torch.tesor): old log probabilities
       error (torch.tensor): can be any error, td_erro, GAE, baseline
       clip (float): distance from one for clipping
    """
    is_weight = torch.exp(logp - logp_old)  # importance sampling
    error_norm = (error - error.mean()) / (error.std() + 1e-8)
    loss = -error_norm * is_weight
    if clip > 0.0:
        loss = torch.max(
            -error_norm * torch.clamp(is_weight, 1 - clip, 1 + clip), loss
        )
    return loss.mean()


def critic_clip_loss(pred, old_pred, label, clip, critic_loss, reduce=torch.mean):
    """
    Clip critic predictions and calculate loss in regards to labels.

    Args:
        pred (torch.tensor): predicted values
        old_pred (torch.tensor): old predicted values
        label (torch.tensor): labels
        clip (float): clip value
        critic_loss (torch.tensor): critic loss to calculate loss between pred and label
    """
    loss = critic_loss(pred, label)
    if clip > 0.0:
        pred_clipped = old_pred + torch.clamp(pred - old_pred, -clip, clip)
        loss = torch.max(critic_loss(pred_clipped, label), loss)
    return reduce(loss)


def discounted_returns(returns, dones, gamma):
    """
    Return for each step eq: dr_t = r_t + γ * r_(t+1) + γ^2 * r_(t+2) + ...
    It is important that the returns include full trajectories, otherwise the discounted
    return will not be correct since the end of the trajectory would be missing.

    Args:
        returns (torch.tensor): returns in order for each step
        dones (torch.tensor): booleans indicating if trajectory is finished
        gamma (float): discount factor
    """
    disc_ret, terminal = returns, 1 - dones
    for i, step_ret in enumerate(torch.flip(returns[:-1], [0])):
        disc_ret[- i - 2] = step_ret + terminal[- i - 2] * gamma * disc_ret[- i - 1]
    return disc_ret.flatten()


def get_prob_at_pos(val_heatmap, pos):
    """
    Get probabilities in the heatmap at the given positions.

    Args:
        val_heatmap (torch.tensor): heatmaps
        pos (torch.tensor): positions corresposning to each heatmap
    """
    assert len(val_heatmap) == len(pos)
    pixel_pos_prob = torch.empty(len(pos)).to(device=val_heatmap.device)
    for i, h in enumerate(val_heatmap):
        if len(pos.shape) == 2:
            pixel_pos_prob[i] = h[0, 0, int(pos[i, 0]), int(pos[i, 1])]
        else:
            pixel_pos_prob[i] = h[0, 0, int(pos[i])]
    return pixel_pos_prob
